"""Aggregate checks on extracted TN3K and DDTI packages. Reads image bytes without running repository code."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(a):
    h=hashlib.sha256(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def quantiles(vals):
    return None if not vals else [round(float(x),6) for x in np.quantile(vals,[0,.1,.5,.9,1])]
def audit(images,masks,label):
    def filemap(folder):
        return {p.stem:p for p in folder.iterdir() if not p.name.startswith('.') and p.suffix.lower() in {'.png','.jpg','.jpeg'}}
    im,ma=filemap(images),filemap(masks)
    ims=set(im); ms=set(ma); cases=sorted(ims&ms)
    hist=Counter();sizes=Counter();bad=Counter();iq=[];dq=[];frac=[];gray=Counter()
    hashes=defaultdict(set); pairs=Counter(); binary_pairs=Counter(); dh={}
    for k in cases:
        with Image.open(im[k]) as f: a=np.array(f.convert('RGB'))
        with Image.open(ma[k]) as f: m=np.array(f.convert('L'))
        ih=identity(a)
        if a.shape[:2]!=m.shape:bad['different_image_mask_dimensions']+=1
        sizes[f'{a.shape[1]}x{a.shape[0]}']+=1
        mx=int(m.max());hist[mx]+=1
        if not np.array_equal(a[:,:,0],a[:,:,1]) or not np.array_equal(a[:,:,1],a[:,:,2]):gray['has_nonidentical_RGB_channels']+=1
        if not mx:bad['empty_raw_mask']+=1
        if np.any((m!=0)&(m!=mx)):bad['intermediate_gray_levels']+=1
        half=m.astype('f4')>mx/2
        extreme=(m==mx)&(mx>0)
        fixed=m>127
        if not fixed.any():bad['empty_if_fixed_gt127']+=1
        if half.any():
            dice=2*np.count_nonzero(half&extreme)/(half.sum()+extreme.sum())
            dq.append(dice); iq.append(float(np.mean(half!=extreme)));frac.append(float(np.mean(half)))
        mh=identity(half); hashes[ih].add(mh);pairs[ih,mh]+=1
        binary_pairs[ih,identity(m)]+=1
        dh[k]=(ih,mh)
    duplicate_keys={k:len(v) for k,v in hashes.items() if len(v)>1}
    return dict(label=label,images=len(im),masks=len(ma),paired=len(cases),unmatched_image=len(ims-ms),
        unmatched_mask=len(ms-ims),unique_decoded_RGB=len(hashes),unique_RGB_and_decoded_L_mask=len(binary_pairs),
        unique_RGB_and_halfmax_mask=len(pairs),identical_RGB_groups_with_different_halfmax_masks=len(duplicate_keys),
        shape_count=len(sizes),common_sizes=sizes.most_common(3),mask_maxima=dict(hist),checks=dict(bad),channels=dict(gray),
        max_only_vs_halfmax_dice=quantiles(dq),max_only_vs_halfmax_fraction_pixels_changed=quantiles(iq),
        foreground_area_fraction=quantiles(frac)),dh

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tn3k',type=Path,required=True);p.add_argument('--ddti',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    root=args.tn3k
    a,h1=audit(root/'trainval-image',root/'trainval-mask','TN3K trainval')
    b,h2=audit(root/'test-image',root/'test-mask','TN3K official test, file QA only')
    c,h3=audit(args.ddti/'img',args.ddti/'label','DDTI via US-Segmentator')
    sets=[{pair[0] for pair in d.values()} for d in (h1,h2,h3)]
    ddnames=[f for f in (args.ddti/'img').glob('*.PNG.png') if f.name[:-8] in h3]
    same_alias=sum(identity(np.array(Image.open(f).convert('RGB')))==h3[f.name[:-8]][0] for f in ddnames)
    lists={s:set((args.ddti/'MainPatient'/(s+'.txt')).read_text().splitlines()) for s in ('train','val','test')}
    assert set.union(*lists.values())==set(h3)
    out=dict(stats=[a,b,c],cross_split_shared_decoded_RGB=len(sets[0]&sets[1]),
       shared_decoded_RGB_between_TN3K_and_DDTI=len((sets[0]|sets[1])&sets[2]),
       ddti_double_extension_pairs=len(ddnames),ddti_double_extension_identical_RGB=int(same_alias),
       ddti_split_sizes={s:len(v) for s,v in lists.items()},
       ddti_train_test_shared_decoded_RGB=len({h3[k][0] for k in lists['train']}&{h3[k][0] for k in lists['test']}),
       quantile_order=['min','p10','median','p90','max'])
    args.out.write_text(json.dumps(out,indent=2)+'\n')
