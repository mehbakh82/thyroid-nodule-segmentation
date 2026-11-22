"""Generate the local two-page supervisor discussion brief from paired development records."""
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from review import overlay

def build(data,run):
    records={r['i']:r for r in json.loads((run/'paired.json').read_text())}
    picks=dict(json.loads((run/'panel-cases.json').read_text()))
    images=np.load(data/'images.npy',mmap_mode='r');gt=np.load(data/'masks.npy',mmap_mode='r')
    pred={}
    for arm in ('gray','edge'):
        with np.load(run/arm/'predictions.npz') as z:
            pred[arm]={int(i):np.unpackbits(a,axis=-1)[:,:images.shape[-1]].astype(bool) for i,a in zip(z['ids'],z['bits'])}
    summary=json.loads((run/'summary.json').read_text())
    overall=summary['overall']
    c=Canvas(str(run/'meeting-brief.pdf'),pagesize=(600,810))
    c.setTitle('Thyroid nodule segmentation: matched development experiment')
    def line(x,y,s,sz=11,bold=False,color='#20333D'):
        c.setFillColor(HexColor(color));c.setFont('Helvetica-Bold' if bold else 'Helvetica',sz);c.drawString(x,y,s)
    def row(name,y):
        i=picks[name];r=records[i]
        line(25,y,name,11,True)
        line(207,y,f"Dice  {r['gray']['dice']:.3f}",11)
        line(394,y,f"Dice  {r['edge']['dice']:.3f}",11)
        views=[Image.fromarray(images[i])]+[overlay(images[i],gt[i].astype(bool),pred[a][i]) for a in ('gray','edge')]
        for left,img in zip((25,211,397),views):
            c.drawImage(ImageReader(img),left,y-153,width=146,height=146)
    def columns(y):
        line(25,y,'Ultrasound',10,True);line(211,y,'Grayscale U-Net',10,True);line(397,y,'Contrast + gradient U-Net',10,True)
        line(25,y-16,'Cyan: reference contour     Amber: prediction     White: shared contour pixels',9)
    line(25,767,'Does adding edge information help on TN3K?',22,True)
    line(25,741,'Matched development experiment for discussion with Dr. Mina Amiri',13)
    line(25,714,'2,303 training images and 576 validation images. Both models trained for 12 epochs.',11)
    lo,hi=overall['conditional_image_interval']
    line(25,694,f"Mean Dice {overall['gray']['dice']:.4f} to {overall['edge']['dice']:.4f}. Paired change {100*overall['mean_delta']:+.2f} points (95% image interval {100*lo:+.2f} to {100*hi:+.2f}).",10.5)
    line(25,675,'One seed and one image-level split. Multiple images can belong to the same patient.',10)
    columns(646)
    for name,y in zip(('Rescue','Harm','Both miss'),(600,423,246)):row(name,y)
    line(25,53,'Which is more consequential in the intended task: missing the target, selecting another',11,True)
    line(25,36,'structure, or imprecisely tracing an already localized nodule?',11,True)
    c.showPage()
    line(25,767,'Where the contour changes',22,True)
    weak=summary['groups']['weak'];small=summary['groups']['small']
    line(25,739,f"Low-gradient proxy ({weak['n']} images): {100*weak['mean_delta']:+.2f} Dice points. Small nodules ({small['n']}): {100*small['mean_delta']:+.2f} points.",11)
    line(25,719,'The low-gradient subgroup uses reference contours for analysis. It is a candidate for',10.5)
    line(25,702,'comparison with clinical ratings of poorly defined margins.',10.5)
    columns(666)
    for name,y in zip(('Boundary improves','Boundary worsens','Extra foreground increases'),(617,437,257)):row(name,y)
    line(25,73,'Examples are selected near the median Dice change within each displayed error category.',9)
    line(25,54,'TN3K development images: Gong et al.   |   Distances use a 224-pixel long-side grid.',9)
    line(25,37,'Source: doi.org/10.1016/j.compbiomed.2022.106389',9)
    c.showPage();c.save()

if __name__=='__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2]))
