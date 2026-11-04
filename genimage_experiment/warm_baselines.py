import json
from pathlib import Path
import timm
import torch

torch.set_num_threads(4)
records = {}
for name in ['resnet50', 'deit_small_patch16_224', 'swin_tiny_patch4_window7_224']:
    print('Loading pretrained', name, flush=True)
    model = timm.create_model(name, pretrained=True, num_classes=0)
    records[name] = model.pretrained_cfg
    del model
Path('baseline_pretrained_sources.json').write_text(json.dumps(records, indent=2))
print('All baseline weights ready', flush=True)
