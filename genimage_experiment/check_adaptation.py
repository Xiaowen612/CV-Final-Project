"""Confirm similar CLIP/Effort scores are not due to accidentally frozen adapters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

checkpoint = torch.load('runs_pilot/effort/seed_42/best.pt', map_location='cpu', weights_only=False)
state = checkpoint['model']
changes = []
for key, value in state.items():
    if key.endswith('.self_attn.q_proj.u') or any(key.endswith(f'.self_attn.{p}.u') for p in ['k_proj','v_proj','out_proj']):
        changes.append(float(torch.linalg.norm(value-state[key[:-1]+'u0'])))
assert len(changes)==96 and min(changes)>0
clip = pd.read_csv('runs_pilot/clip_linear/seed_42/predictions/clean.csv').set_index('path')
effort = pd.read_csv('runs_pilot/effort/seed_42/predictions/clean.csv').set_index('path')
delta = np.abs(clip.probability-effort.loc[clip.index].probability)
result=dict(status='passed',attention_projections_updated=len(changes),
            min_u_change_norm=min(changes),max_u_change_norm=max(changes),
            mean_absolute_probability_difference=float(delta.mean()),
            max_absolute_probability_difference=float(delta.max()),
            interpretation='Adapters changed, but this short rank-1 pilot produced very similar predictions to the frozen baseline.')
Path('adaptation_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
