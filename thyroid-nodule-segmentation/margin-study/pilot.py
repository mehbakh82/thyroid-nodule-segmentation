"""Cross-fitted local contour correction, with fixed controls and alternative references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate import measure,sha
from boundary_audit import score

METHODS=('baseline','active_contour','single_head','dual_mean','selective')
DECODERS=('above127','only255')


def signed_band(mask):
    if not mask.any() or mask.all():
        return np.zeros_like(mask),np.zeros_like(mask,dtype=float)
    sd=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
    return np.abs(sd)<=4,sd


def describe(image,prob,disagreement):
    mask=prob>=.5
    band,sd=signed_band(mask)
    if not band.any():
        return band,np.empty((0,10),dtype='f4')
    im=image.astype('f4')/255
    mean=ndi.gaussian_filter(im,2)
    std=np.maximum(ndi.gaussian_filter(im*im,2)-mean*mean,0)**.5
    grad=ndi.gaussian_gradient_magnitude(im,1)
    fine=ndi.gaussian_gradient_magnitude(im,2)
    normals=np.stack(np.gradient(ndi.gaussian_filter(sd,1)))[:,band]
    normals/=np.maximum(np.linalg.norm(normals,axis=0),1e-6)
    coords=np.array(np.where(band))
    samples=[ndi.map_coordinates(im,coords+d*normals,order=1,mode='nearest') for d in (2,-2)]
    return band,np.column_stack([v[band] for v in (prob,disagreement,sd,im,mean,std,grad,fine)]+samples).astype('f4')


def classifiers(X,y,weights):
    out=[]
    for k in range(2):
        clf=HistGradientBoostingClassifier(max_iter=70,learning_rate=.08,max_leaf_nodes=15,
            min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=241123)
        clf.fit(X,y[:,k],sample_weight=weights)
        out.append(clf)
    return out


def apply_heads(initial,band,pa,pb):
    conservative=initial[band].copy()
    conservative[(pa>=.7)&(pb>=.7)]=True
    conservative[(pa<=.3)&(pb<=.3)]=False
    results={}
    for name,label in [('single_head',pa>=.5),('dual_mean',(pa+pb)/2>=.5),('selective',conservative)]:
        m=initial.copy()
        m[band]=label
        results[name]=m
    return results


def classical(im,mask,band):
    if not band.any():return mask.copy()
    indicator=inverse_gaussian_gradient(im.astype('f4')/255,alpha=100,sigma=1)
    m=morphological_geodesic_active_contour(indicator,4,init_level_set=mask.astype('int8'),
                                           smoothing=1,threshold='auto',balloon=0)
    return np.where(band,m.astype(bool),mask)


def load_inputs(masks):
    data=ROOT/'local/data'
    manifest=json.loads((data/'manifest.json').read_bytes())
    work=ROOT/'local/margin-study'
    receipt=json.loads((work/'probabilities.json').read_bytes())
    assert receipt['manifest_sha256']==sha(data/'manifest.json')
    assert receipt['array_sha256']==sha(work/'probabilities.npy')
    for name,expected in manifest['arrays'].items():assert sha(data/name)==expected
    images=np.load(data/'images.npy',mmap_mode='r');gt=np.load(data/'masks.npy',mmap_mode='r')
    probs=np.load(work/'probabilities.npy',mmap_mode='r')
    rows=[r for r in manifest['rows'] if r['role']=='validation']
    assert receipt['ids']==[r['i'] for r in rows] and len(rows)==576
    cases=[]
    for j,r in enumerate(rows):
        idx=r['i'];x,y,w,h=r['box']; crop=np.s_[y:y+h,x:x+w]
        im=images[idx][crop];p,dis=[q[crop] for q in probs[j]]
        band,features=describe(im,p,dis)
        path=masks/r['file'];assert sha(path)==r['mask_sha256']
        with Image.open(path) as f:raw=np.asarray(f.convert('L'))
        refs=[np.array(Image.fromarray((raw>th).astype('uint8')).resize((w,h),Image.Resampling.NEAREST),bool) for th in (127,254)]
        np.testing.assert_array_equal(gt[idx][crop],refs[0])
        assert all(t.any() for t in refs)
        cases.append(dict(i=idx,name=r['file'],im=im,base=p>=.5,X=features,band=band,refs=refs,
                          Y=np.column_stack([t[band] for t in refs])))
    return cases,receipt


def fit_cases(cases):
    blocks=[];labels=[];weights=[]
    for c in cases:
        if len(c['X'])==0:continue
        sample=np.random.default_rng([241123,c['i']]).choice(len(c['X']),min(len(c['X']),192),replace=False)
        blocks.append(c['X'][sample]);labels.append(c['Y'][sample])
        weights.extend([1/len(sample)]*len(sample))
    w=np.asarray(weights);return classifiers(np.concatenate(blocks),np.concatenate(labels),w/w.mean())


def evaluate(pred,base,refs):
    stable=refs[0]==refs[1];changed=pred!=base
    stats=dict(edits=int(changed.sum()),fixed=int((stable&changed&(pred==refs[0])).sum()),
               damaged=int((stable&changed&(pred!=refs[0])).sum()),ambiguous_edits=int((~stable&changed).sum()))
    assert stats['edits']==stats['fixed']+stats['damaged']+stats['ambiguous_edits']
    for k,t in zip(DECODERS,refs):
        d,s=score(pred,t)
        m=measure(pred,t)
        np.testing.assert_allclose([m['dice'],m['surface_dice']],[d,s[1]],atol=1.e-9)
        stats[k]=dict(dice=d,s1=s[0],s2=s[1],s3=s[2],hd95=m['hd95'],miss=m['miss'])
    return stats


def delta(rows,method,control,decoder,metric):
    d=np.array([r[method][decoder][metric]-r[control][decoder][metric] for r in rows])
    rng=np.random.default_rng(241123)
    means=d[rng.integers(len(rows),size=(5000,len(rows)))].mean(axis=1)
    return dict(mean_points=float(100*d.mean()),ci_points=(100*np.quantile(means,[.025,.975])).tolist(),
                up_five=int((d>=.05).sum()),down_five=int((d<=-.05).sum()))


def summarize(rows):
    local=[r for r in rows if r['baseline']['above127']['dice']>=.5]
    out={}
    for name in METHODS:
        summary={}
        for k in DECODERS:
            summary[k]=dict(whole_mean_dice=float(np.mean([r[name][k]['dice'] for r in rows])),
                whole_mean_surface2=float(np.mean([r[name][k]['s2'] for r in rows])),
                whole_mean_hd95=float(np.mean([r[name][k]['hd95'] for r in rows])),
                whole_misses=sum(r[name][k]['miss'] for r in rows),
                localized_mean_surface2=float(np.mean([r[name][k]['s2'] for r in local])),
                by_tolerance={m:delta(local,name,'baseline',k,m) for m in ('s1','s2','s3')})
        summary['stable_edits']={m:sum(r[name][m] for r in rows) for m in ('fixed','damaged','ambiguous_edits','edits')}
        summary['localized_stable_edits']={m:sum(r[name][m] for r in local) for m in ('fixed','damaged','ambiguous_edits','edits')}
        out[name]=summary
    candidate=out['selective'];primary=candidate['above127']['by_tolerance']['s2']
    edits=candidate['localized_stable_edits']
    passed=(primary['mean_points']>=2 and primary['ci_points'][0]>0 and
            candidate['only255']['by_tolerance']['s2']['mean_points']>0 and edits['fixed']>edits['damaged'])
    return dict(images=len(rows),localized=len(local),methods=out,primary=primary,
                selective_minus_dual=delta(local,'selective','dual_mean','above127','s2'),
                decision='advance' if passed else 'stop this rule')


def main(masks):
    out=ROOT/'local/margin-study';cases,receipt=load_inputs(masks)
    order=sorted(range(len(cases)),key=lambda j:hashlib.sha256(('margin-correction-v1/'+cases[j]['name']).encode()).digest())
    folds={j:rank%5 for rank,j in enumerate(order)}
    original=json.loads((ROOT/'runs/ablation-fold0/fusion-audit-paired.json').read_bytes())
    prior={r['i']:r['scores']['gray_flip_average'] for r in original}
    rows=[]
    for fold in range(5):
        training=[c for j,c in enumerate(cases) if folds[j]!=fold]
        testing=[c for j,c in enumerate(cases) if folds[j]==fold]
        assert not {c['i'] for c in training}&{c['i'] for c in testing}
        heads=fit_cases(training)
        for c in testing:
            base,band=c['base'],c['band']
            scores=[clf.predict_proba(c['X'])[:,1] if band.any() else np.empty(0) for clf in heads]
            predictions=apply_heads(base,band,*scores)
            predictions.update(baseline=base,active_contour=classical(c['im'],base,band))
            row=dict(i=c['i'],fold=fold)
            for name,pred in predictions.items():
                assert not np.any((base!=pred)&~band)
                row[name]=evaluate(pred,base,c['refs'])
            original_score=prior[c['i']]
            np.testing.assert_allclose([row['baseline']['above127']['dice'],row['baseline']['above127']['s2']],
                                       [original_score['dice'],original_score['surface_dice']],atol=1e-10)
            rows.append(row)
        print('scored fold',fold,len(testing),flush=True)
    rows.sort(key=lambda r:r['i'])
    assert len(set(r['i'] for r in rows))==len(rows)==576
    per_image=out/'per-image.json';per_image.write_text(json.dumps(rows,allow_nan=False)+'\n')
    result=summarize(rows)|dict(per_image_sha256=sha(per_image),checkpoint_sha256=receipt['checkpoint_sha256'],
        pilot_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
        fold_sizes=[sum(r['fold']==k for r in rows) for k in range(5)])
    (ROOT/'margin-study/results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['decision'],result['primary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trainval_masks',type=Path)
    main(p.parse_args().trainval_masks)
