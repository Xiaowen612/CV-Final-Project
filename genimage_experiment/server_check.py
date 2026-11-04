"""Verify real pretrained CLIP and one Effort optimization step on GPU."""
import json
import time
from pathlib import Path
import torch
import yaml
from experiment.models import Detector

torch.set_num_threads(4)
cfg = yaml.safe_load(Path('config_pilot.yaml').read_text())
cfg['clip_id'] = '/home/smbu/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
Path('config_server_pilot.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
start = time.perf_counter()
model = Detector(cfg, 'effort').cuda().train()
print('Model initialized', time.perf_counter()-start, flush=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
x = torch.randn(2, 3, 224, 224, device='cuda')
y = torch.tensor([0., 1.], device='cuda')
with torch.autocast('cuda', dtype=torch.float16):
    logits = model(x)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
penalty = model.regularization(cfg)
loss = bce+penalty
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
optimizer.step()
torch.cuda.synchronize()
info = dict(status='passed', scope='Random inputs test GPU execution ONLY, not detection performance',
            torch=torch.__version__, gpu=torch.cuda.get_device_name(), seconds=time.perf_counter()-start,
            loss=float(loss.detach()), bce=float(bce.detach()), penalty=float(penalty.detach()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_memory_mb=torch.cuda.max_memory_allocated()/2**20)
Path('gpu_check.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info), flush=True)
