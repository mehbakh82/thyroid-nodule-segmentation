"""Independently reconcile pre-meeting evidence. Uses files and CPU arithmetic only."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def read(p):return json.loads(p.read_bytes())


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root):
    evidence=root.parent/'msc-readiness/thyroid-evidence'
    sources=read(evidence/'sources.json')['checked_files']
    for name,rec in sources.items():assert digest(evidence/'local-data'/name)==rec['sha256']
    data=root/'local/data';meta=read(data/'manifest.json')
    base=root/'runs/source-fold0'; extra=root/'runs/ablation-fold0'
    original=read(base/'run.json');second=read(extra/'run.json')
    assert original['manifest']==digest(data/'manifest.json')
    assert all(original[k]==second[k] for k in original)
    for name,expected in meta['arrays'].items():assert digest(data/name)==expected
    for name,expected in [('model.py',original['model']),('train.py',original['trainer']),('channel_ablation.py',second['ablation'])]:
        assert digest(root/name)==expected
    val=[r['i'] for r in meta['rows'] if r['role']=='validation']
    assert len(set(val))==576
    truth=np.load(data/'masks.npy',mmap_mode='r')
    factorial=read(root/'results/channel-ablation.json')
    scores={}
    for arm,run in [('gray',base),('edge',base),('contrast',extra),('gradient',extra)]:
        folder=run/arm;receipt=read(folder/'evaluation.json')
        assert receipt==factorial['receipts'][arm]
        assert digest(folder/'checkpoint.pt')==receipt['checkpoint_sha256']
        assert digest(folder/'predictions.npz')==receipt['predictions_sha256']
        assert read(run/'replay.json')[arm]==576
        with np.load(folder/'predictions.npz') as z:
            np.testing.assert_array_equal(z['ids'],val)
            vals=[]
            for row,bits in zip(val,z['bits'],strict=True):
                target=truth[row].astype(bool)
                pred=np.unpackbits(bits,axis=1)[:,:target.shape[1]].astype(bool)
                vals.append(2*np.count_nonzero(pred&target)/(pred.sum()+target.sum()))
        scores[arm]=float(np.mean(vals))
        np.testing.assert_allclose(scores[arm],factorial['overall']['means'][arm]['dice'],rtol=0,atol=1e-12)
    assert len({x['initial_sha256'] for x in factorial['receipts'].values()})==1
    assert factorial['paired_sha256']==digest(extra/'paired-four.json')
    fusion=read(root/'results/fusion-audit.json');pairfile=extra/'fusion-audit-paired.json'
    assert fusion['script_sha256']==digest(root/'fusion_audit.py') and digest(pairfile)==fusion['per_image_sha256']
    pairs=read(pairfile);assert [r['i'] for r in pairs]==val
    for method,v in fusion['methods'].items():
        for m in ('dice','surface_dice','hd95'):
            np.testing.assert_allclose(v[m],np.mean([r['scores'][method][m] for r in pairs]))
    cases=extra/'contrast-case-review'
    key=read(cases/'key.json')
    assert len({r['i'] for r in key})==len(key)==12 and sum(r['A']=='gray' for r in key)==6
    with (cases/'response-sheet.csv').open(newline='') as f:
        answers=list(csv.DictReader(f))
    assert {int(r['case']) for r in answers}=={r['case'] for r in key}
    return dict(training_images=original['train'],validation_images=len(val),independent_recomputed_dice=scores,
                linked_archive_digests=len(sources),verified_checkpoints=4,verified_saved_binary_masks=4*len(val),
                fusion_rows=len(pairs),blinded_cases=len(key),reader_response_rows=len(answers),
                ddti_map_sha256=digest(root/'local/ddti-preflight/image-reference-map.json'))


if __name__=='__main__':
    root=Path(__file__).resolve().parent;receipt=verify(root)
    assert receipt['ddti_map_sha256']==read(root/'results/ddti-preflight.json')['manifest_sha256']
    (root/'results/stage-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)
