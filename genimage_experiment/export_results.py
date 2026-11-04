"""Export code, manifests, logs and measured results, excluding data/weights/env."""
from pathlib import Path
import tarfile

root = Path('.')
include_roots = {'experiment', 'tests', 'manifests_pilot', 'logs', 'runs_pilot'}
include_suffixes = {'.py', '.yaml', '.md', '.json', '.csv', '.png', '.txt', '.ini', '.sh', '.log'}
with tarfile.open('experiment_results.tar.gz', 'w:gz') as tar:
    provenance = Path('data/GenImage/download_provenance.json')
    if provenance.exists():
        tar.add(provenance, arcname=provenance.as_posix())
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        parts = path.parts
        if len(parts) > 1 and parts[0] not in include_roots:
            continue
        if '__pycache__' in parts or path.suffix not in include_suffixes:
            continue
        if path.name in {'project.tar.gz', 'experiment_results.tar.gz'}:
            continue
        tar.add(path, arcname=path.as_posix())
print(Path('experiment_results.tar.gz').stat().st_size)
