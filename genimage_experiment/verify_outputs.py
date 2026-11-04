"""Recompute saved metrics and verify all methods used the same test images."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from experiment.metrics import metrics
from experiment.data import read_manifest
from experiment.engine import fingerprint

cfg = yaml.safe_load(Path('config_server_repeated.yaml').read_text())
expected = pd.DataFrame(read_manifest(Path(cfg['manifest_dir'])/'test.csv')).set_index('path')
all_runs = sorted(Path(cfg['run_dir']).glob('*/seed_*/run.json'))
assert len(all_runs) == 24, f'Expected 24 trained models, got {len(all_runs)}'
checks = 0
for path in all_runs:
    meta = json.loads(path.read_text())
    assert meta['manifest_hashes'] == fingerprint(cfg)
    scores = pd.read_csv(path.parent/'metrics.csv')
    assert len(scores) == 70
    assert scores.threshold.nunique() == 1
    for condition in cfg['conditions']:
        pred = pd.read_csv(path.parent/'predictions'/f'{condition}.csv').set_index('path')
        assert pred.index.is_unique and set(pred.index) == set(expected.index)
        assert (pred.label == expected.loc[pred.index].label).all()
        assert (pred.generator == expected.loc[pred.index].generator).all()
        assert ((pred.probability >= 0) & (pred.probability <= 1)).all()
        groups = {g:[g] for g in cfg['seen']+cfg['unseen']}
        groups.update(seen=cfg['seen'],unseen=cfg['unseen'],all=cfg['seen']+cfg['unseen'])
        for group, generators in groups.items():
            row = scores[(scores.condition==condition)&(scores.group==group)].iloc[0]
            sub = pred[pred.generator.isin(generators)]
            actual = metrics(sub.label,sub.probability,float(row.threshold))
            for metric,value in actual.items():
                assert np.isclose(value,row[metric],atol=1e-9), (path,condition,group,metric)
            checks += 1
result = dict(status='passed',trained_models=len(all_runs),recomputed_metric_groups=checks,
              prediction_rows=24*len(cfg['conditions'])*len(expected),
              test_images=len(expected),shared_manifests=True,test_threshold_retuning=False)
Path('results_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
