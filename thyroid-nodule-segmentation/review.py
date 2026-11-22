"""Produce local-only image review sheets. No scores are used to select the caliper panel."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def selected(rows):
    return sorted((r for r in rows if r['role']=='validation'),key=lambda r:
                  hashlib.sha256(('thyroid-caliper-review-v1/'+r['file']).encode()).digest())[:36]


def blind(data,out):
    rows=selected(json.loads((data/'manifest.json').read_text())['rows'])
    im=np.load(data/'images.npy',mmap_mode='r')
    out.mkdir(parents=True,exist_ok=True)
    (out/'blind-order.json').write_text(json.dumps([r['i'] for r in rows]))
    for k in range(3):
        canvas=Image.new('RGB',(4*224,3*246),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[k*12:k*12+12]):
            x,y=(j%4)*224,(j//4)*246
            canvas.paste(Image.fromarray(im[r['i']]),(x,y))
            draw.text((x+4,y+227),f'{k*12+j+1}',fill='black')
        canvas.save(out/f'blind-{k+1}.png')


def overlay(im,truth,pred):
    a=np.repeat(im[:,:,None],3,axis=2).copy()
    def boundary(mask):
        frame=np.pad(mask,1,constant_values=False)
        interior=mask & frame[1:-1,:-2] & frame[1:-1,2:] & frame[:-2,1:-1] & frame[2:,1:-1]
        return mask ^ interior
    b=boundary(truth)
    p=boundary(pred)
    a[b]=[0,210,255];a[p]=[255,148,25];a[b&p]=[255,255,255]
    return Image.fromarray(a)

def comparison(data,run):
    cases=json.loads((run/'paired.json').read_text())
    im=np.load(data/'images.npy',mmap_mode='r');gt=np.load(data/'masks.npy',mmap_mode='r')
    preds={}
    for arm in ('gray','edge'):
        with np.load(run/arm/'predictions.npz') as r:
            preds[arm]={int(i):np.unpackbits(m,axis=-1)[:,:im.shape[-1]].astype(bool)
                        for i,m in zip(r['ids'],r['bits'])}
    used=set()
    def representative(name,condition):
        avail=[c for c in cases if condition(c) and c['i'] not in used]
        if not avail:return []
        mid=np.median([c['delta'] for c in avail])
        c=min(avail,key=lambda c:(abs(c['delta']-mid),c['i']))
        used.add(c['i'])
        return [(name,c)]
    examples=[]
    for group,check in [('Rescue',lambda c:c['rescue']),('Harm',lambda c:c['harm']),
            ('Both miss',lambda c:c['gray']['miss'] and c['edge']['miss']),
            ('Boundary improves',lambda c:c['gray']['dice']>=.5 and c['edge']['dice']>=.5 and c['edge']['surface_dice']-c['gray']['surface_dice']>=.1),
            ('Boundary worsens',lambda c:c['gray']['dice']>=.5 and c['edge']['dice']>=.5 and c['edge']['surface_dice']-c['gray']['surface_dice']<=-.1),
            ('Extra foreground increases',lambda c:c['edge']['fp']>c['gray']['fp'] and c['delta']<-.05),
            ('Little change',lambda c:abs(c['delta'])<.01),
            ('Visible marker, gain',lambda c:c.get('marker')=='clear' and c['delta']>0),
            ('Visible marker, loss',lambda c:c.get('marker')=='clear' and c['delta']<0)]:
        examples.extend(representative(group,check))
    canvas=Image.new('RGB',(800,56+245*len(examples)),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),'TN3K development: cyan = reference, amber = prediction, white = shared boundary',fill='black')
    for x,label in [(10,'Source'),(250,'Grayscale'),(490,'Grayscale + contrast + gradient')]:draw.text((x,33),label,fill='black')
    for j,(label,r) in enumerate(examples):
        i=r['i'];y=55+245*j
        canvas.paste(Image.fromarray(im[i]),(10,y))
        for x,arm in [(250,'gray'),(490,'edge')]:
            canvas.paste(overlay(im[i],gt[i].astype(bool),preds[arm][i]),(x,y))
        draw.text((10,y+226),f"{label}   |   {r['file']}   |   Dice {r['gray']['dice']:.3f} to {r['edge']['dice']:.3f}",fill='black')
    canvas.save(run/'meeting-panel.png')
    (run/'panel-cases.json').write_text(json.dumps([(name,c['i']) for name,c in examples]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['blind','comparison']);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dir',type=Path,required=True);a=p.parse_args()
    (blind if a.stage=='blind' else comparison)(a.data,a.dir)
