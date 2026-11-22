"""Render the blinded review packet from the sampling key written by boundary_audit.py."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from review import overlay
from review_records import prepare_sheet


def packet(root, out=None, sources=None, appearance_first=False):
    local=root/'local/data'
    run=root/'runs/source-fold0'
    if out is None:out=run/'boundary-audit'
    if sources is None:sources={arm:run/arm/'predictions.npz' for arm in ('gray','edge')}
    key=json.loads((out/'key.json').read_bytes())
    im=np.load(local/'images.npy',mmap_mode='r')
    masks={}
    for arm,path in sources.items():
        with np.load(path) as z:
            masks[arm]={int(i):np.unpackbits(b,axis=-1)[:,:im.shape[-1]].astype(bool)
                        for i,b in zip(z['ids'],z['bits'])}
    prepare_sheet(out/'response-sheet.csv',key,appearance_first)
    c=Canvas(str(out/'blinded-cases.pdf'),pagesize=(720,790))
    c.setTitle('Thyroid nodule contour review')
    def text(x,y,s,size=10,bold=False):
        c.setFillColor(HexColor('#20343a'));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        c.drawString(x,y,s)
    if appearance_first:
        for start in range(0,len(key),6):
            text(22,754,'First pass: review the ultrasound alone',21,True)
            text(22,729,'Before viewing the contours, record target clarity, margin visibility and any measurement marks.',12)
            for j,r in enumerate(key[start:start+6]):
                x=22+230*(j%3); y=681-318*(j//3)
                text(x,y,f"Case {r['case']:02}",14,True)
                c.drawImage(ImageReader(Image.fromarray(im[r['i']])),x,y-219,width=204,height=204)
                text(x,y-237,'Target clear?   Yes / No / Unsure',10)
                text(x,y-254,'Margin:   Clear / Mixed / Indistinct',10)
                text(x,y-271,'Calipers:   Present / Absent / Unsure',10)
            c.showPage()
    for page in range(0,len(key),2):
        text(22,754,'Thyroid nodule contour review',21,True)
        text(22,731,'Inspect the ultrasound first. Choose the contour you would retain for the intended task.',12)
        text(22,713,'A and B are randomized for each case. Mark uncertain when the target or margin is ambiguous.',10)
        for n,r in enumerate(key[page:page+2]):
            y=680-318*n
            i=r['i'];first=r['A'];second=next(k for k in masks if k!=first)
            text(22,y,f"Case {r['case']:02}",14,True)
            raw=im[i]; blank=np.zeros_like(raw,bool)
            images=[Image.fromarray(raw),overlay(raw,blank,masks[first][i]),overlay(raw,blank,masks[second][i])]
            for x,label,img in zip([22,257,492],['Ultrasound','Contour A','Contour B'],images):
                text(x,y-18,label,10)
                c.drawImage(ImageReader(img),x,y-228,width=204,height=204)
            text(22,y-246,'Preference     A  /  B  /  equivalent  /  neither acceptable  /  uncertain',10,True)
            text(22,y-265,'Main concern     wrong target  /  missed tissue  /  excess tissue  /  ambiguous margin  /  none  /  other',10)
            text(22,y-285,'Consequential correction and location: ____________________________________________________________',10)
        text(22,22,'TN3K development examples, Gong et al.  |  Case discussion sample  |  Scores and model identities held in a separate key.',9)
        c.showPage()
    c.save()


if __name__=='__main__':
    packet(Path(__file__).resolve().parent)
