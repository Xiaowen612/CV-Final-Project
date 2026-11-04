"""Deterministic manifests, global content deduplication and image transforms."""
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path):
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['label'] = int(row['label'])
    return rows


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def source_id(path, label):
    # Prevent separately encoded copies of a known ImageNet image crossing splits.
    match = re.search(r'(ILSVRC2012_val_\d+|n\d{8}_\d+)', path.stem)
    return match.group(1) if label == 0 and match else ''


def prepare(cfg):
    root, output = Path(cfg['data_root']).resolve(), Path(cfg['manifest_dir'])
    if (output / 'audit.json').exists():
        raise FileExistsError('Manifests already exist. Use a NEW manifest_dir to change the split.')
    rng = random.Random(cfg['split_seed'])
    used_hash, used_id = set(), set()
    rows, rejected = [], []
    generators = cfg['seen'] + cfg['unseen']
    if len(generators) != len(set(generators)):
        raise ValueError('Seen and unseen generators must be disjoint and unique.')
    # Reserve final-test content first; any identical training content is then excluded.
    tasks = [(g, 'val', 'test', cfg['test_per_class']) for g in generators]
    tasks += [(g, 'train', 'train_dev', cfg['train_per_class'] + cfg['dev_per_class'])
              for g in cfg['seen']]
    for generator, official_split, split, count in tasks:
        for label, folder in [(0, 'nature'), (1, 'ai')]:
            directory = root / cfg['generators'][generator] / official_split / folder
            if not directory.is_dir():
                raise FileNotFoundError(f'Missing GenImage directory: {directory}')
            files = sorted(p for p in directory.rglob('*') if p.suffix.lower() in EXTENSIONS)
            rng.shuffle(files)
            accepted = []
            for path in files:
                if len(accepted) == count:
                    break
                try:
                    with Image.open(path) as original:
                        image = ImageOps.exif_transpose(original).convert('RGB')
                        image.load()
                        digest = pixel_hash(image)
                        width, height = image.size
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    rejected.append({'path': str(path), 'reason': type(error).__name__})
                    continue
                identity = source_id(path, label)
                if digest in used_hash or (identity and identity in used_id):
                    rejected.append({'path': str(path), 'reason': 'duplicate_content_or_imagenet_id'})
                    continue
                used_hash.add(digest)
                if identity:
                    used_id.add(identity)
                accepted.append(dict(path=path.relative_to(root).as_posix(), label=label,
                                     generator=generator, split=split, official_split=official_split,
                                     sha256=digest, source_id=identity, width=width, height=height))
            if len(accepted) != count:
                raise ValueError(f'{directory}: need {count} unique valid images, found {len(accepted)}. '
                                 'Reduce sample counts or provide more data; no silent shortfall.')
            if split == 'train_dev':
                for i, row in enumerate(accepted):
                    row['split'] = 'dev' if i < cfg['dev_per_class'] else 'train'
            rows.extend(accepted)
            print(f'{generator}/{official_split}/{folder}: {len(accepted)} selected', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        write_csv(output / f'{split}.csv', [r for r in rows if r['split'] == split])
    write_csv(output / 'rejected.csv', rejected, ['path', 'reason'])
    audit = {'config': cfg, 'counts': dict(Counter(f"{r['split']}/{r['generator']}/{r['label']}" for r in rows)),
             'global_unique_pixels': len(used_hash), 'selected': len(rows),
             'rejected_scanned': len(rejected),
             'scope': 'All SELECTED images checked globally; unsampled files are not exhaustively audited.',
             'near_duplicates': 'Not detected beyond exact RGB pixels and recognized ImageNet source IDs.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')


def degrade(image, condition):
    if condition == 'clean':
        return image
    if condition.startswith('jpeg'):
        with io.BytesIO() as stream:
            image.save(stream, format='JPEG', quality=int(condition[4:]), subsampling=2)
            stream.seek(0)
            with Image.open(stream) as decoded:
                return decoded.convert('RGB').copy()
    if condition.startswith('blur'):
        array = np.asarray(image).astype(np.float32)
        array = gaussian_filter(array, sigma=(float(condition[4:]), float(condition[4:]), 0))
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if condition.startswith('scale'):
        ratio = float(condition[5:])
        if not 0 < ratio <= 1:
            raise ValueError('Scale must be in (0, 1].')
        size = image.size
        return image.resize(tuple(max(1, round(v * ratio)) for v in size), Image.Resampling.BICUBIC).resize(size, Image.Resampling.BICUBIC)
    raise ValueError(f'Unknown condition: {condition}')


def transform(image, cfg, method, training=False, condition='clean'):
    # Test degradations act on original RGB pixels, before model resize/crop.
    image = degrade(image, condition)
    short = cfg['resize_short']
    w, h = image.size
    image = image.resize((round(w * short / min(w, h)), round(h * short / min(w, h))), Image.Resampling.BICUBIC)
    if training:
        if method in {'effort_aug', 'cnnspot'}:
            if random.random() < 0.5:
                image = degrade(image, f'blur{random.uniform(0, 3):.5f}')
            if random.random() < 0.5:
                image = degrade(image, f'jpeg{random.randint(30, 100)}')
            if method == 'effort_aug' and random.random() < 0.5:
                image = degrade(image, f'scale{random.uniform(0.25, 1):.5f}')
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
    n = cfg['image_size']
    if training:
        left, top = random.randint(0, image.width - n), random.randint(0, image.height - n)
    else:
        left, top = (image.width - n) // 2, (image.height - n) // 2
    image = image.crop((left, top, left+n, top+n))
    array = np.asarray(image, dtype=np.float32).copy() / 255
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    is_clip = method in {'clip_linear', 'clip_full', 'effort', 'effort_aug', 'effort_no_orth', 'effort_no_energy', 'effort_no_constraints'}
    mean, std = (CLIP_MEAN, CLIP_STD) if is_clip else (IMAGENET_MEAN, IMAGENET_STD)
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class Images(Dataset):
    def __init__(self, rows, cfg, method, training=False, condition='clean'):
        self.rows, self.cfg, self.method = rows, cfg, method
        self.training, self.condition = training, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(Path(self.cfg['data_root']) / row['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            tensor = transform(image, self.cfg, self.method, self.training, self.condition)
        return tensor, row['label'], index
