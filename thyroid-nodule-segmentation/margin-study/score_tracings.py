"""Validate an independent TN3K tracing return and compare it with frozen references."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from statistics import mean
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boundary_audit import score
from evaluate import sha
from tracing_packet import sha as bytes_sha


def load_return(folder, cases):
    with (folder / 'observations.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'case', 'status', 'margin_visibility', 'measurement_marks'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Use the observations.csv headers supplied in the kit.')
        rows = list(reader)
    ids = [r['case'] for r in rows]
    if len(ids) != len(cases) or set(ids) != set(cases):
        raise ValueError('Expected one observations row per supplied case ID.')
    masks = {p.name.removesuffix('-mask.png'): p for p in folder.glob('*-mask.png')}
    if masks.keys() - set(cases):
        raise ValueError('Unexpected mask case IDs.')
    for row in rows:
        status = row['status'].strip().lower() or 'pending'
        if status not in {'traced', 'target_unclear', 'untraceable', 'pending'}:
            raise ValueError('Unrecognized status for '+row['case'])
        if (row['case'] in masks) != (status == 'traced'):
            raise ValueError('Mask and traced status must agree for '+row['case'])
        row['status'] = status
    return rows, masks


def reader_mask(path, size):
    with Image.open(path) as im:
        if im.format != 'PNG' or im.mode != 'L' or im.size != size:
            raise ValueError(f'{path.name}: expected an 8-bit grayscale PNG of native size {size}')
        a = np.asarray(im)
    if set(np.unique(a)) != {0, 255}:
        raise ValueError(f'{path.name}: expected background 0 and nonempty foreground 255')
    return a == 255


def resize_mask(mask, size):
    im = Image.fromarray(mask.astype('uint8'))
    scaled = np.asarray(im.resize(size, Image.Resampling.NEAREST), dtype=bool)
    if not scaled.any():
        raise ValueError('A nonempty mask disappeared on the evaluation grid; inspect before scoring.')
    return scaled


def compare(raw, reader, baseline):
    if raw.shape != reader.shape or raw.max() != 255:
        raise ValueError('Source and reader masks must share native dimensions and the expected encoding.')
    refs = {'above127': raw > 127, 'only255': raw == 255}
    agreement = {}
    for name, ref in refs.items():
        agreement[name] = dict(dice=float(2*(reader & ref).sum()/(reader.sum()+ref.sum())),
                               relative_area_error=float(ref.sum()/reader.sum()-1))
    shrunk = ndi.binary_erosion(baseline)
    grid = {}
    for name, mask in {**refs, 'reader': reader}.items():
        target = resize_mask(mask, baseline.shape[::-1])
        a, s = score(baseline, target)
        b, t = score(shrunk, target)
        grid[name] = dict(region_dice=float(a), shrunk_region_dice=float(b), surface2=float(s[1]),
                          shrunk_surface2=float(t[1]), delta_surface2_points=float(100*(t[1]-s[1])))
    return dict(native_agreement=agreement, evaluation_grid=grid)


def summarize(results):
    usable = [r for r in results if r['status'] == 'traced']
    out = dict(cases=len(results), dispositions=dict(Counter(r['status'] for r in results)),
               traced_by_size=dict(Counter(r['stratum'] for r in usable)))
    if usable:
        out['mean_native_dice'] = {k:mean(r['native_agreement'][k]['dice'] for r in usable)
                                   for k in ('above127','only255')}
        out['mean_native_relative_area_error'] = {k:mean(r['native_agreement'][k]['relative_area_error'] for r in usable)
                                                  for k in ('above127','only255')}
        out['mean_erosion_delta_points'] = {k:mean(r['evaluation_grid'][k]['delta_surface2_points'] for r in usable)
                                            for k in ('above127','only255','reader')}
        def sign(d):
            return 'gain' if d>1e-9 else 'loss' if d< -1e-9 else 'tie'
        out['reader_erosion_directions'] = dict(Counter(sign(r['evaluation_grid']['reader']['delta_surface2_points']) for r in usable))
        out['opposite_erosion_direction_vs_reader'] = {k:sum(
            {sign(r['evaluation_grid'][k]['delta_surface2_points']), sign(r['evaluation_grid']['reader']['delta_surface2_points'])} == {'gain','loss'}
            for r in usable) for k in ('above127','only255')}
    return out


def analyze(folder, source):
    work = ROOT/'local/margin-study'
    kit = work/'independent-contour-review'
    keyfile = kit/'private-key.json'
    mapping = json.loads(keyfile.read_bytes())
    keyed = {r['case']:r for r in mapping}
    if len(mapping) != 12 or len(keyed) != 12 or len({r['index'] for r in mapping}) != 12:
        raise ValueError('Unexpected case key')
    observations, masks = load_return(folder, keyed)
    receipt = json.loads((work/'probabilities.json').read_bytes())
    manifest_path = ROOT/'local/data/manifest.json'
    prob_path = work/'probabilities.npy'
    if sha(manifest_path) != receipt['manifest_sha256'] or sha(prob_path) != receipt['array_sha256']:
        raise ValueError('Frozen prediction or development manifest mismatch')
    if receipt['checkpoint_sha256'] != json.loads((ROOT/'margin-study/results.json').read_bytes())['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint mismatch')
    manifest = json.loads(manifest_path.read_bytes())['rows']
    positions = {idx:j for j,idx in enumerate(receipt['ids'])}
    probs = np.load(prob_path, mmap_mode='r')
    rows = []
    with ZipFile(kit/'reader-kit.zip') as archive:
        for row in observations:
            name = row['case']
            k = keyed[name]
            meta = manifest[k['index']]
            png = archive.read('images/'+name+'.png')
            if meta['file'] != k['source'] or meta['role'] != 'validation' or bytes_sha(png) != k['export_sha256']:
                raise ValueError('Kit or source mapping mismatch for '+name)
            rec = dict(case=name, status=row['status'], stratum=k['size_stratum'],
                       margin_visibility=row['margin_visibility'], measurement_marks=row['measurement_marks'])
            if rec['status'] == 'traced':
                with Image.open(io.BytesIO(png)) as image:
                    target = reader_mask(masks[name], image.size)
                path = source/meta['file']
                if sha(path) != meta['mask_sha256']:
                    raise ValueError('Source mask mismatch for '+name)
                with Image.open(path) as im:
                    raw = np.asarray(im.convert('L'))
                x,y,w,h = meta['box']
                baseline = probs[positions[k['index']],0,y:y+h,x:x+w] >= .5
                rec.update(compare(raw,target,baseline), reader_mask_sha256=sha(masks[name]))
            rows.append(rec)
    return dict(summary=summarize(rows),cases=rows,kit_sha256=sha(kit/'reader-kit.zip'),
                private_key_sha256=sha(keyfile),observations_sha256=sha(folder/'observations.csv'),
                probability_sha256=receipt['array_sha256'],development_manifest_sha256=receipt['manifest_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('return_folder',type=Path);p.add_argument('trainval_masks',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.return_folder,args.trainval_masks)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with args.out.open('x') as out:out.write(text)
    print(json.dumps(result['summary'],indent=2))
