import torch
from PIL import Image, ImageOps
from .data import transform
from .engine import load_checkpoint, device_for


def infer(checkpoint_path, image_path, device='auto'):
    target = device_for({'device': device})
    model, checkpoint = load_checkpoint(checkpoint_path, target)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        x = transform(image, checkpoint['config'], checkpoint['method']).unsqueeze(0).to(target)
    with torch.no_grad():
        probability = float(model(x).sigmoid().item())
    return {'label': 'AI-Generated' if probability >= checkpoint['threshold'] else 'Real',
            'ai_probability': probability, 'threshold': checkpoint['threshold'],
            'note': 'Model score, not a calibrated probability or proof of image provenance.'}
