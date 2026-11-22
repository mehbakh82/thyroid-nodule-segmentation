"""Verify model-run receipts and recompute the headline development scores using the Python standard library."""
import json
import math
from pathlib import Path
from statistics import fmean
from zipfile import ZipFile

from restore_checkpoints import ARMS, PROJECT, checksum


def read(path):
    return json.loads(path.read_bytes())


def check_records(path, expected_digest, expected_ids):
    if checksum(path) != expected_digest:
        raise ValueError(f'Digest mismatch: {path}')
    records = read(path)
    if [row['i'] for row in records] != expected_ids:
        raise ValueError(f'Unexpected image IDs or row order: {path}')
    return records


def verify():
    results = PROJECT / 'results'
    runs = PROJECT / 'runs'
    base = runs / 'source-fold0'
    ablation = runs / 'ablation-fold0'
    manifest = PROJECT / 'local/data/manifest.json'
    spec = read(base / 'run.json')
    if checksum(manifest) != spec['manifest']:
        raise ValueError('Input manifest differs from the training manifest')
    ids = [row['i'] for row in read(manifest)['rows'] if row['role'] == 'validation']
    if len(set(ids)) != len(ids) or len(ids) != spec['val']:
        raise ValueError('Validation denominator mismatch')
    report = read(results / 'channel-ablation.json')
    rows = check_records(ablation / 'paired-four.json', report['paired_sha256'], ids)
    for fold, arms in ARMS.items():
        for arm in arms:
            folder = runs / fold / arm
            receipt = read(folder / 'evaluation.json')
            predictions = folder / 'predictions.npz'
            if receipt != report['receipts'][arm] or checksum(predictions) != receipt['predictions_sha256']:
                raise ValueError(f'Prediction receipt mismatch for {arm}')
            with ZipFile(predictions) as archive:
                if archive.testzip() or set(archive.namelist()) != {'ids.npy', 'bits.npy'}:
                    raise ValueError(f'Invalid prediction archive for {arm}')
            score = fmean(row[arm]['dice'] for row in rows)
            if not math.isclose(score, report['overall']['means'][arm]['dice'], abs_tol=1e-12):
                raise ValueError(f'Per-image Dice mismatch for {arm}')
            checkpoint = folder / 'checkpoint.pt'
            if checkpoint.exists() and checksum(checkpoint) != receipt['checkpoint_sha256']:
                raise ValueError(f'Checkpoint mismatch for {arm}')
    fusion = read(results / 'fusion-audit.json')
    pairs = check_records(ablation / 'fusion-audit-paired.json', fusion['per_image_sha256'], ids)
    for name, metrics in fusion['methods'].items():
        for metric in ('dice', 'surface_dice', 'hd95'):
            score = fmean(row['scores'][name][metric] for row in pairs)
            if not math.isclose(score, metrics[metric], abs_tol=1e-12):
                raise ValueError(f'Per-image {metric} mismatch for {name}')
    print(f'Verified four prediction archives and paired aggregate scores for {len(ids)} validation images.')


if __name__ == '__main__':
    verify()
