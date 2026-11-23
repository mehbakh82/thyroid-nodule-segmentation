"""Prepare an image-only native-resolution re-tracing kit for a small, blinded development audit."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(namespace, filename):
    return sha((namespace+'/'+filename).encode())


def build(image_dir):
    data=ROOT/'local/data'
    run=ROOT/'runs/source-fold0'
    margin=ROOT/'local/margin-study'
    paired=margin/'per-image.json'
    assert sha(paired.read_bytes())==json.loads((ROOT/'margin-study/results.json').read_bytes())['per_image_sha256']
    assert sha((data/'manifest.json').read_bytes())==json.loads((margin/'probabilities.json').read_bytes())['manifest_sha256']
    manifest=json.loads((data/'manifest.json').read_bytes())
    eligible={r['i'] for r in json.loads(paired.read_bytes()) if r['baseline']['above127']['dice']>=.50}
    exclude=set(json.loads((ROOT/'local/blind/blind-order.json').read_bytes()))
    exclude.update(j for _,j in json.loads((run/'panel-cases.json').read_bytes()))
    for path in [run/'boundary-audit/key.json',ROOT/'runs/ablation-fold0/contrast-case-review/key.json']:
        exclude.update(r['i'] for r in json.loads(path.read_bytes()))
    pool=[r for r in manifest['rows'] if r['role']=='validation' and r['i'] in eligible-exclude]
    cutoff=manifest['subgroup']['area_q25']
    chosen=[]
    for small in (True,False):
        available=[r for r in pool if (r['area']<=cutoff)==small]
        chosen.extend(sorted(available,key=lambda r:rank('trace-audit-v1',r['file']))[:6])
    chosen.sort(key=lambda r:rank('trace-order-v1',r['file']))
    assert len({r['i'] for r in chosen})==12
    out=margin/'independent-contour-review';out.mkdir(exist_ok=True)
    packet=out/'reader-kit.zip';key=[]
    instructions="""TN3K thyroid nodule re-tracing pilot

Please review the 12 images independently, before viewing any original masks, model overlays or other reader's answers. The purpose is to establish which boundaries can be judged reliably. If more than one target is plausible, record target_unclear. If there is a plausible target but insufficient evidence to outline it, record untraceable. Otherwise mark traced and delineate the nodule as you would for the intended task. Please note any arcs where a defensible alternative contour is possible.

Each PNG preserves the native pixel grid and appearance. Use your usual annotation tool and return a single-channel, lossless binary PNG per traced case (background 0, nodule 255), with the same dimensions and alignment as the image. Name outputs case-01-mask.png, etc. If exporting polygons instead, return the native-coordinate polygon and specify origin/axes and tool. Record visibility (clear, mixed, indistinct), visible measurement marks (yes, no, unsure), and any target ambiguity in the CSV. Do not infer physical distances from pixel counts.

These are selected research cases for a discussion sample. Images originate from the TN3K author release (Gong et al., doi:10.1016/j.compbiomed.2022.106389).
"""
    sheet=io.StringIO(newline='')
    writer=csv.writer(sheet);writer.writerow(['case','status','margin_visibility','measurement_marks','uncertain_arc_or_target','notes'])
    with ZipFile(packet,'w',ZIP_DEFLATED) as z:
        z.writestr('instructions.txt',instructions)
        for j,r in enumerate(chosen,1):
            name=f'case-{j:02}'
            original=(image_dir/r['file']).read_bytes()
            assert sha(original)==r['image_sha256']
            with Image.open(io.BytesIO(original)) as im:
                # Fresh grayscale image preserves the displayed pixels and drops embedded JPEG metadata.
                g=im.convert('L')
                clean=Image.frombytes('L',g.size,g.tobytes())
                b=io.BytesIO();clean.save(b,format='PNG')
                png=b.getvalue()
                with Image.open(io.BytesIO(png)) as roundtrip:
                    assert roundtrip.size==g.size and roundtrip.tobytes()==g.tobytes()
                z.writestr('images/'+name+'.png',png)
            key.append(dict(case=name,source=r['file'],index=r['i'],export_sha256=sha(png),
                            size_stratum='small' if r['area']<=cutoff else 'other'))
            writer.writerow([name,'','','','',''])
        z.writestr('observations.csv',sheet.getvalue())
    (out/'private-key.json').write_text(json.dumps(key,indent=2)+'\n')
    with ZipFile(packet) as z:
        assert z.testzip() is None and len(z.namelist())==14
    print(packet,len(chosen),len(pool))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trainval_images',type=Path)
    build(p.parse_args().trainval_images)
