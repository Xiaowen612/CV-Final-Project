import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

from experiment.data import prepare, read_manifest, degrade
from experiment.metrics import metrics, select_threshold
from experiment.models import SVDResidualLinear, Detector


def test_svd_reconstruction_and_gradients():
    torch.manual_seed(7)
    original = torch.nn.Linear(12, 12)
    layer = SVDResidualLinear(original, rank=3)
    x = torch.randn(2, 5, 12)
    torch.testing.assert_close(layer(x), original(x), atol=2e-6, rtol=2e-5)
    layer(x).square().mean().backward()
    for param in [layer.u, layer.s, layer.v]:
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert not layer.main.requires_grad


def test_constraints_equal_full_matrix_reference():
    torch.manual_seed(9)
    original = torch.nn.Linear(12, 12)
    layer = SVDResidualLinear(original, rank=2)
    u, _, vh = torch.linalg.svd(original.weight.detach())
    with torch.no_grad():
        layer.u.add_(torch.randn_like(layer.u)*.03)
        layer.v.add_(torch.randn_like(layer.v)*.03)
        layer.s.add_(.1)
    orth, energy = layer.constraints()
    full_u = torch.cat([u[:, :-2], layer.u], dim=1)
    full_v = torch.cat([vh[:-2], layer.v], dim=0)
    expected = .5*torch.linalg.norm(full_u.T@full_u-torch.eye(12)) + .5*torch.linalg.norm(full_v@full_v.T-torch.eye(12))
    expected_energy = ((layer.main+(layer.u*layer.s)@layer.v).square().sum()-original.weight.square().sum()).abs()
    torch.testing.assert_close(orth, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(energy, expected_energy, atol=2e-5, rtol=2e-5)
    (orth+energy).backward()
    assert torch.isfinite(layer.u.grad).all()


def test_threshold_and_metrics():
    y, p = [0, 0, 1, 1, 0, 1], [.1, .4, .3, .9, .2, .8]
    threshold = select_threshold(y, p)
    best = max(balanced_accuracy_score(y, np.array(p)>=t) for t in p)
    assert metrics(y, p, threshold)['balanced_accuracy'] == best
    perfect = metrics([0, 0, 1, 1], [.1, .2, .8, .9], .5)
    assert perfect['roc_auc'] == 1 and perfect['real_fpr'] == 0 and perfect['tp'] == 2


def test_degradations_deterministic():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8))
    for condition in ['clean', 'jpeg90', 'jpeg70', 'jpeg50', 'blur0.5', 'blur1', 'blur2', 'scale0.75', 'scale0.5', 'scale0.25']:
        a, b = degrade(image, condition), degrade(image, condition)
        assert a.size == image.size and a.tobytes() == b.tobytes()


def test_manifest_no_leakage_and_official_splits(tmp_path):
    cfg = dict(data_root=str(tmp_path/'data'), manifest_dir=str(tmp_path/'manifests'),
               generators={'s': 'S', 'u': 'U'}, seen=['s'], unseen=['u'], split_seed=3,
               train_per_class=4, dev_per_class=2, test_per_class=3)
    rng = np.random.default_rng(10)
    for generator in ['S', 'U']:
        for split in ['train', 'val']:
            for category in ['ai', 'nature']:
                folder = Path(cfg['data_root'])/generator/split/category
                folder.mkdir(parents=True)
                for i in range(12):
                    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(folder/f'{i}.png')
    # Exact duplicate across official splits must not leak.
    source = Path(cfg['data_root'])/'S/val/ai/0.png'
    (Path(cfg['data_root'])/'S/train/ai/dup.png').write_bytes(source.read_bytes())
    prepare(cfg)
    rows = {s: read_manifest(Path(cfg['manifest_dir'])/f'{s}.csv') for s in ['train', 'dev', 'test']}
    hashes = [{r['sha256'] for r in rows[s]} for s in rows]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    assert all(r['generator']=='s' and r['official_split']=='train' for r in rows['train']+rows['dev'])
    assert all(r['official_split']=='val' for r in rows['test'])
    second = dict(cfg, manifest_dir=str(tmp_path/'manifests2'))
    prepare(second)
    assert (Path(cfg['manifest_dir'])/'train.csv').read_bytes() == (Path(second['manifest_dir'])/'train.csv').read_bytes()


def test_clip_adapter_checkpoint_roundtrip_and_backward():
    cfg = dict(gradient_checkpointing=True, residual_rank=1, lambda_orth=1., lambda_energy=1.)
    spec = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14)
    # Tiny RANDOM CLIP is only an architecture test, never a benchmark substitute.
    model = Detector(cfg, 'effort', pretrained=False, model_spec=spec)
    for layer in model.backbone.vision_model.encoder.layers:
        for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            setattr(layer.self_attn, name, SVDResidualLinear(torch.nn.Linear(16, 16), 1))
    model.train()
    loss = model(torch.randn(2, 3, 28, 28)).square().mean()+model.regularization(cfg)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    clone = Detector(cfg, 'effort', pretrained=False, model_spec=spec)
    clone.load_state_dict(model.state_dict())
    model.eval(); clone.eval()
    x = torch.randn(2, 3, 28, 28)
    torch.testing.assert_close(model(x), clone(x))


def test_end_to_end_train_evaluate_report(tmp_path):
    from experiment.engine import train, evaluate
    from experiment.report import report
    from experiment.data import write_csv
    cfg = dict(data_root=str(tmp_path/'data'), manifest_dir=str(tmp_path/'manifests'),
               run_dir=str(tmp_path/'runs'), image_size=16, resize_short=20,
               seen=['s'], unseen=['u'], train_per_class=3, dev_per_class=2, test_per_class=2,
               epochs=2, patience=2, batch_size=2, eval_batch_size=2, accumulation_steps=2,
               lr=.001, head_lr=.001, weight_decay=0., workers=0, device='cpu', amp=False,
               lambda_orth=1., lambda_energy=1., conditions=['clean','jpeg50','blur1','scale0.5'])
    rng = np.random.default_rng(0)
    for split in ['train', 'dev', 'test']:
        rows = []
        for g in (['s', 'u'] if split == 'test' else ['s']):
            for label in [0, 1]:
                for i in range(3 if split == 'train' else 2):
                    path = f'{split}_{g}_{label}_{i}.png'
                    directory = Path(cfg['data_root']); directory.mkdir(exist_ok=True)
                    Image.fromarray(rng.integers(0,256,(24,24,3),dtype=np.uint8)).save(directory/path)
                    rows.append(dict(path=path,label=label,generator=g,official_split='val' if split=='test' else 'train'))
        write_csv(Path(cfg['manifest_dir'])/f'{split}.csv',rows)
    checkpoint = train(cfg, 'tiny_smoke', 1)
    results = evaluate(cfg, checkpoint)
    assert len(results) == 20
    report(cfg)
    assert (Path(cfg['run_dir'])/'report/RESULTS.md').is_file()
