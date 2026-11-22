"""Blinded discussion sample for the contrast-only versus grayscale follow-up."""
import hashlib
import json
from pathlib import Path
from review_records import save_key

import numpy as np

from reader_packet import packet


def build(root):
    reference=root/'runs/source-fold0'
    run=root/'runs/ablation-fold0'
    out=run/'contrast-case-review'
    out.mkdir(exist_ok=True)
    rows=json.loads((run/'paired-four.json').read_bytes())
    receipt=json.loads((root/'results/channel-ablation.json').read_bytes())
    assert hashlib.sha256((run/'paired-four.json').read_bytes()).hexdigest()==receipt['paired_sha256']
    sources={'gray':reference/'gray/predictions.npz','contrast':run/'contrast/predictions.npz'}
    for arm,p in sources.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['receipts'][arm]['predictions_sha256']
    exclude={r['i'] for r in json.loads((reference/'boundary-audit/key.json').read_bytes())}
    exclude.update(i for name,i in json.loads((reference/'panel-cases.json').read_bytes()))
    rng=np.random.default_rng(241124)
    groups={'contrast higher':[], 'grayscale higher':[], 'similar':[]}
    for r in rows:
        if r['i'] in exclude:continue
        gain=r['contrast']['dice']-r['gray']['dice']
        if gain>=.05:groups['contrast higher'].append(r['i'])
        elif gain<=-.05:groups['grayscale higher'].append(r['i'])
        elif abs(gain)<.01:groups['similar'].append(r['i'])
    sample=[]
    for label,ids in groups.items():
        selected=rng.choice(sorted(ids),size=4,replace=False)
        assignment=['gray','gray','contrast','contrast']
        rng.shuffle(assignment)
        sample.extend(dict(i=int(i),stratum=label,A=a) for i,a in zip(selected,assignment))
    rng.shuffle(sample)
    for case,r in enumerate(sample,1):r['case']=case
    save_key(out,sample)
    packet(root,out,sources,appearance_first=True)


if __name__=='__main__':build(Path(__file__).resolve().parent)
