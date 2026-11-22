"""Render a local one-page summary using examples fixed by the original two-arm study."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from review import overlay

ARMS=('gray','contrast','gradient','edge')
NAMES=('Grayscale','+ contrast','+ gradient','+ both')


def build():
    prior=Path('runs/source-fold0');new=Path('runs/ablation-fold0')
    result=json.loads(Path('results/channel-ablation.json').read_bytes())['overall']
    examples=dict(json.loads((prior/'panel-cases.json').read_bytes()))
    cases={r['i']:r for r in json.loads((new/'paired-four.json').read_bytes())}
    images=np.load('local/data/images.npy',mmap_mode='r');targets=np.load('local/data/masks.npy',mmap_mode='r')
    masks={}
    for arm in ARMS:
        with np.load((prior if arm in ('gray','edge') else new)/arm/'predictions.npz') as f:
            masks[arm]={int(i):np.unpackbits(b,axis=-1)[:,:images.shape[-1]].astype(bool) for i,b in zip(f['ids'],f['bits'])}
    c=Canvas(str(new/'meeting-addendum.pdf'),pagesize=(690,735))
    c.setTitle('TN3K development: contrast and gradient ablation')
    def text(x,y,s,size=11,bold=False,color='#21323e'):
        c.setFillColor(HexColor(color));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,y,s)
    text(24,696,'Do contrast and gradients help separately?',22,True)
    text(24,671,'TN3K source development  |  2,303 training images, 576 validation images  |  one paired seed',11)
    cols=[24,172,270,409,539]
    for x,s in zip(cols,['Input','Mean Dice','Surface Dice (2 px)','Complete misses','Gain vs gray (pp)']):
        text(x,633,s,10,True)
    for j,(arm,name) in enumerate(zip(ARMS,NAMES)):
        row=result['means'][arm];y=610-21*j
        gain=(row['dice']-result['means']['gray']['dice'])*100
        for x,s in zip(cols,[name,f"{row['dice']:.4f}",f"{row['surface_dice']:.4f}",str(row['zero_overlap']),f'{gain:+.2f}' if j else 'Reference']):
            text(x,y,s)
    text(24,510,'Same cases shown at the first comparison',14,True)
    text(24,491,'Cyan traces the reference, amber the prediction, and white their shared contour pixels.',10)
    positions=[24,157,290,423,556]
    for x,s in zip(positions,['Ultrasound',*NAMES]):text(x,463,s,10,True)
    for j,category in enumerate(['Rescue','Harm']):
        i=examples[category];r=cases[i];top=450-j*175
        ims=[Image.fromarray(images[i])]+[overlay(images[i],targets[i].astype(bool),masks[a][i]) for a in ARMS]
        for x,img in zip(positions,ims):
            c.drawImage(ImageReader(img),x,top-110,width=110,height=110)
        text(positions[0],top-128,r['file'],10)
        for x,a in zip(positions[1:],ARMS):text(x,top-128,f"Dice {r[a]['dice']:.3f}",10)
    text(24,94,'The two examples were selected from the earlier grayscale-versus-combined results before the new fits.',10)
    text(24,74,'Uncertainty estimates in the results file are image-level and conditional on these four trained models.',10)
    text(24,42,'Source images: TN3K, Gong et al.  |  doi.org/10.1016/j.compbiomed.2022.106389',9,color='#546774')
    c.showPage();c.save()


if __name__=='__main__':build()
