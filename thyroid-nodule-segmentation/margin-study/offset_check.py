"""Post-hoc diagnostic. Apply a fixed one-pixel change without looking at CT/US appearance."""
import argparse,json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from pilot import ROOT,DECODERS,score,sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('masks',type=Path)
    mask_dir=p.parse_args().masks
    work=ROOT/'local/margin-study'
    receipt=json.loads((work/'probabilities.json').read_bytes())
    probs=np.load(work/'probabilities.npy',mmap_mode='r')
    data=ROOT/'local/data';manifest=json.loads((data/'manifest.json').read_bytes())
    assert sha(work/'probabilities.npy')==receipt['array_sha256']
    out={k:{name:[] for name in ('baseline','shrink_one','grow_one')} for k in DECODERS}
    removed=enclosed=deep=with_holes=0
    filled_change={k:[] for k in DECODERS}
    for j,i in enumerate(receipt['ids']):
        r=manifest['rows'][i];x,y,w,h=r['box']
        assert r['role']=='validation' and sha(mask_dir/r['file'])==r['mask_sha256']
        with Image.open(mask_dir/r['file']) as f:raw=np.asarray(f.convert('L'))
        base=probs[j,0,y:y+h,x:x+w]>=.5
        preds=dict(baseline=base,shrink_one=ndi.binary_erosion(base),grow_one=ndi.binary_dilation(base))
        for k,th in zip(DECODERS,(127,254)):
            gt=np.array(Image.fromarray((raw>th).astype('uint8')).resize((w,h),Image.Resampling.NEAREST),bool)
            d,s=score(base,gt)
            if k==DECODERS[0] and d<.5:break
            if k==DECODERS[0]:
                primary=gt
            else:
                lost=primary & ~gt
                interior=lost & ndi.binary_fill_holes(gt)
                removed+=int(lost.sum())
                enclosed+=int(interior.sum())
                deep+=int((lost & (ndi.distance_transform_edt(primary)>4)).sum())
                with_holes+=int(interior.any())
            out[k]['baseline'].append(float(s[1]))
            filled=ndi.binary_fill_holes(gt)
            filled_change[k].append(float(score(preds['shrink_one'],filled)[1][1]-score(base,filled)[1][1]))
            for name in ('shrink_one','grow_one'):
                out[k][name].append(float(score(preds[name],gt)[1][1]))
    stats={}
    for k,values in out.items():
        b=np.asarray(values['baseline'])
        stats[k]=dict(n=len(b),delta_surface_dice_points={n:float(100*np.mean(np.array(values[n])-b)) for n in ('shrink_one','grow_one')})
    assert stats['above127']['n']==stats['only255']['n']
    stats['decoding_geometry']=dict(removed_foreground_pixels=removed,
        removed_now_enclosed_background=enclosed,removed_farther_than_four_pixels_from_background=deep,
        localized_images_with_newly_enclosed_background=with_holes)
    stats['shrink_one_delta_after_filling_reference_holes_points']={k:100*float(np.mean(v)) for k,v in filled_change.items()}
    (Path(__file__).parent/'offset-check.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(stats)
