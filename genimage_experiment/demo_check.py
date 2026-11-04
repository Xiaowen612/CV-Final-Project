import json
from pathlib import Path
import yaml
from experiment.data import read_manifest
from experiment.inference import infer

cfg = yaml.safe_load(Path('config_server_pilot.yaml').read_text())
row = read_manifest(Path(cfg['manifest_dir'])/'test.csv')[0]
checkpoint = Path(cfg['run_dir'])/'effort/seed_42/best.pt'
result = infer(checkpoint, Path(cfg['data_root'])/row['path'], 'cuda')
result.update(source_path=row['path'], actual_label=row['label'], selection='First manifest row; not cherry-picked')
Path('demo_prediction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
