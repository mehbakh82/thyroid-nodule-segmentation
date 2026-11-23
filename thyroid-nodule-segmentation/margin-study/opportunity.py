"""Diagnostic upper bound for *region* Dice from reference-directed in-band edits."""
from pathlib import Path
import json
import numpy as np
from pilot import ROOT,signed_band,sha


def dice(mask,target):
    return 2*np.count_nonzero(mask&target)/(mask.sum()+target.sum())


if __name__=='__main__':
    data=ROOT/'local/data';work=ROOT/'local/margin-study'
    receipt=json.loads((work/'probabilities.json').read_bytes())
    assert sha(work/'probabilities.npy')==receipt['array_sha256']
    manifest=json.loads((data/'manifest.json').read_bytes());targets=np.load(data/'masks.npy',mmap_mode='r')
    probs=np.load(work/'probabilities.npy',mmap_mode='r')
    rows=[]
    for j,i in enumerate(receipt['ids']):
        r=manifest['rows'][i];assert r['role']=='validation'
        x,y,w,h=r['box'];crop=np.s_[y:y+h,x:x+w]
        mask=probs[j,0][crop]>=.5
        gt=targets[i][crop].astype(bool);base=dice(mask,gt)
        band,_=signed_band(mask)
        oracle=dice(np.where(band,gt,mask),gt)
        rows.append([base,oracle,np.count_nonzero(band&(mask!=gt)),np.count_nonzero(mask!=gt)])
    a=np.asarray(rows);out={}
    for k,b in [('all',a),('localized',a[a[:,0]>=.5])]:
        out[k]=dict(n=len(b),baseline_mean_region_dice=float(b[:,0].mean()),oracle_mean_region_dice=float(b[:,1].mean()),
             oracle_region_dice_headroom_points=float(100*np.mean(b[:,1]-b[:,0])),in_band_incorrect_pixels=int(b[:,2].sum()),
             total_incorrect_pixels=int(b[:,3].sum()),fraction_errors_in_band=float(b[:,2].sum()/b[:,3].sum()))
    (Path(__file__).parent/'opportunity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)
