"""Restore the four original research checkpoints from their public GitHub release."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parent / 'thyroid-nodule-segmentation'
BASE = 'https://github.com/mehbakh82/thyroid-nodule-segmentation/releases/download/v0.1.0'
ARMS = {'source-fold0': ('gray', 'edge'), 'ablation-fold0': ('contrast', 'gradient')}


def checksum(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def restore(fold, arm):
    folder = PROJECT / 'runs' / fold / arm
    expected = json.loads((folder / 'evaluation.json').read_bytes())['checkpoint_sha256']
    dest = folder / 'checkpoint.pt'
    if dest.exists():
        if checksum(dest) != expected:
            raise ValueError(f'Existing file has a different checksum: {dest}')
        return
    request = Request(f'{BASE}/{arm}-checkpoint.pt', headers={'User-Agent': 'TN3K-research-replay'})
    temp = None
    try:
        with urlopen(request, timeout=120) as response, NamedTemporaryFile(dir=folder, delete=False) as out:
            temp = Path(out.name)
            for block in iter(lambda: response.read(1024 * 1024), b''):
                out.write(block)
        if checksum(temp) != expected:
            raise ValueError(f'Checksum mismatch for {arm}')
        temp.replace(dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == '__main__':
    for fold, arms in ARMS.items():
        for arm in arms:
            restore(fold, arm)
            print(f'Verified {fold}/{arm}')
