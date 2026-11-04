import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Images, read_manifest, write_csv
from .metrics import metrics, select_threshold
from .models import Detector


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def device_for(cfg):
    device = cfg.get('device', 'auto')
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)


def fingerprint(cfg):
    return {s: hashlib.sha256((Path(cfg['manifest_dir']) / f'{s}.csv').read_bytes()).hexdigest()
            for s in ['train', 'dev', 'test']}


def loader(rows, cfg, method, training=False, condition='clean', seed=42):
    return DataLoader(Images(rows, cfg, method, training, condition),
                      batch_size=cfg['batch_size'] if training else cfg['eval_batch_size'],
                      shuffle=training, num_workers=cfg['workers'], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), pin_memory=device_for(cfg).type == 'cuda')


@torch.no_grad()
def predict(model, data, device, amp=False):
    model.eval()
    labels, probs, indices = [], [], []
    for x, y, index in data:
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            logits = model(x.to(device))
        labels.extend(y.tolist())
        probs.extend(logits.float().sigmoid().cpu().tolist())
        indices.extend(index.tolist())
    return np.array(labels), np.array(probs), indices


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = Detector(checkpoint['config'], checkpoint['method'], pretrained=False, model_spec=checkpoint['model_spec'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint


def train(cfg, method, seed):
    seed_all(seed)
    device = device_for(cfg)
    run = Path(cfg['run_dir']) / method / f'seed_{seed}'
    if (run / 'best.pt').exists():
        raise FileExistsError(f'{run} already contains a checkpoint. Use a different run_dir.')
    train_rows = read_manifest(Path(cfg['manifest_dir']) / 'train.csv')
    dev_rows = read_manifest(Path(cfg['manifest_dir']) / 'dev.csv')
    for rows in [train_rows, dev_rows]:
        assert all(r['generator'] in cfg['seen'] and r['official_split'] == 'train' for r in rows)
    run.mkdir(parents=True, exist_ok=True)
    hashes = fingerprint(cfg)
    start = time.perf_counter()
    model = Detector(cfg, method, pretrained=method != 'tiny_smoke').to(device)
    setup_seconds = time.perf_counter() - start
    total = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
        {'params': model.head.parameters(), 'lr': cfg['head_lr']}], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
    best, stale, history = -math.inf, 0, []
    for epoch in range(cfg['epochs']):
        # Epoch-specific RNG makes augmentation independent of validation data iteration.
        seed_all(seed + epoch)
        train_loader = loader(train_rows, cfg, method, True, seed=seed+epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start, loss_sum, seen = time.perf_counter(), 0.0, 0
        accumulation = cfg['accumulation_steps']
        batches = len(train_loader)
        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.float().to(device)
            group_start = (step // accumulation) * accumulation
            # Sample-weighted accumulation, including final partial minibatch.
            group_size = min(accumulation * cfg['batch_size'], len(train_rows) - group_start * cfg['batch_size'])
            with torch.autocast(device.type, enabled=cfg['amp'] and device.type == 'cuda'):
                logits = model(x)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
            penalty = model.regularization(cfg)
            loss = bce.float() + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {step}')
            scaler.scale(loss * len(y) / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
            if step % 100 == 0:
                print(f'{method} seed={seed} epoch={epoch+1} batch={step+1}/{batches} loss={float(loss.detach()):.5f}', flush=True)
        y, p, _ = predict(model, loader(dev_rows, cfg, method), device, cfg['amp'])
        threshold = select_threshold(y, p)
        validation = metrics(y, p, threshold)
        record = dict(epoch=epoch+1, train_loss=loss_sum/seen, seconds=time.perf_counter()-epoch_start,
                      **{f'dev_{k}': v for k, v in validation.items()})
        history.append(record)
        write_csv(run / 'history.csv', history)
        print(json.dumps(record), flush=True)
        if validation['roc_auc'] > best:
            best, stale = validation['roc_auc'], 0
            checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              model_spec=model.model_spec, config=cfg, method=method, seed=seed,
                              threshold=threshold, epoch=epoch+1, dev_metrics=validation, manifest_hashes=hashes)
            torch.save(checkpoint, run / 'best.pt')
            write_csv(run / 'dev_predictions.csv', [dict(path=r['path'], label=int(a), probability=float(b))
                                                    for r, a, b in zip(dev_rows, y, p)])
        else:
            stale += 1
        scheduler.step()
        if stale >= cfg['patience']:
            break
    versions = {}
    for package in ['torch', 'torchvision', 'transformers', 'timm', 'numpy', 'Pillow', 'scikit-learn']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = dict(method=method, seed=seed, device=str(device), platform=platform.platform(),
                gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                versions=versions, config=cfg, manifest_hashes=hashes,
                trainable_parameters=trainable, stored_parameters_and_buffers=total,
                setup_seconds=setup_seconds, total_seconds=time.perf_counter()-start,
                peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0,
                best_dev_auc=best, smoke_only=method == 'tiny_smoke')
    info['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path('experiment').glob('*.py'))}
    (run / 'run.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return run / 'best.pt'


def evaluate(cfg, checkpoint_path):
    device = device_for(cfg)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint['manifest_hashes'] != fingerprint(cfg):
        raise ValueError('Manifest fingerprints differ from training; test evaluation refused.')
    # Restore training preprocessing, while allowing data to be moved to another host.
    eval_cfg = dict(checkpoint['config'], data_root=cfg['data_root'], device=cfg['device'],
                    eval_batch_size=cfg['eval_batch_size'], workers=cfg['workers'])
    method, seed, threshold = checkpoint['method'], checkpoint['seed'], checkpoint['threshold']
    rows = read_manifest(Path(cfg['manifest_dir']) / 'test.csv')
    assert all(r['official_split'] == 'val' for r in rows)
    run = Path(checkpoint_path).parent
    results = []
    for condition in cfg['conditions']:
        start = time.perf_counter()
        y, p, indices = predict(model, loader(rows, eval_cfg, method, condition=condition), device, cfg['amp'])
        elapsed = time.perf_counter() - start
        predictions = [dict(path=rows[i]['path'], generator=rows[i]['generator'], label=int(a),
                            probability=float(b), prediction=int(b >= threshold), condition=condition)
                       for i, a, b in zip(indices, y, p)]
        write_csv(run / 'predictions' / f'{condition}.csv', predictions)
        groups = {g: [g] for g in cfg['seen'] + cfg['unseen']}
        groups.update(seen=cfg['seen'], unseen=cfg['unseen'], all=cfg['seen']+cfg['unseen'])
        for group, generators in groups.items():
            mask = np.array([r['generator'] in generators for r in predictions])
            result = dict(method=method, seed=seed, condition=condition, group=group,
                          **metrics(y[mask], p[mask], threshold))
            result['condition_seconds_all_images'] = elapsed
            results.append(result)
        print(f'{method} seed={seed}: evaluated {condition}', flush=True)
    write_csv(run / 'metrics.csv', results)
    return results
