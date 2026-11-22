"""One-page vector figure using only aggregate measurements; paths taken from the command line."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor

def write(folder,out):
    values=[]
    for f in sorted(folder.glob('*.jpg')):
        m=np.asarray(Image.open(f).convert('L'))
        a=m>127;b=m==m.max()
        values.append([a.mean(),2*np.count_nonzero(a&b)/(a.sum()+b.sum())])
    v=np.array(values);cuts=np.quantile(v[:,0],[.25,.5,.75]);groups=np.searchsorted(cuts,v[:,0],side='left')
    c=Canvas(str(out),pagesize=(640,465));c.setTitle('TN3K reference-mask decoding by nodule size')
    ink,teal,muted,light=map(HexColor,['#192D37','#14695F','#445B68','#DAE2E5'])
    def label(x,y,s,size=10,font='Helvetica',color=ink):
        c.setFillColor(color);c.setFont(font,size);c.drawString(x,y,s)
    label(32,428,'TN3K reference masks disagree more for small nodules',20,'Helvetica-Bold')
    label(32,402,'Same JPEG, two binary masks: foreground >127 versus foreground exactly 255.',12)
    label(32,381,f'{len(v):,} development images. Each image contributes one Dice score.',10,color=muted)
    left,right=147,592
    def X(d):return left+(d-.55)/.45*(right-left)
    for t in [.6,.7,.8,.9,1.]:
        c.setStrokeColor(light);c.line(X(t),108,X(t),333)
        label(X(t)-9,92,f'{t:.1f}',10,color=muted)
    for i in range(4):
        arr=v[groups==i,1];p10,q1,med,q3,p90=np.quantile(arr,[.1,.25,.5,.75,.9])
        y=305-56*i
        label(32,y+2,['Smallest 25%','Second quarter','Third quarter','Largest 25%'][i],11,'Helvetica-Bold')
        label(32,y-13,f'{len(arr)} images',9,color=muted)
        c.setStrokeColor(teal);c.setLineWidth(1.4);c.line(X(p10),y,X(p90),y)
        for t in [p10,p90]:c.line(X(t),y-4,X(t),y+4)
        c.setFillColor(HexColor('#C8E1DA'));c.rect(X(q1),y-7,X(q3)-X(q1),14,stroke=1,fill=1)
        c.setFillColor(ink);c.circle(X(med),y,3,fill=1,stroke=0);label(X(med)-14,y+12,f'{med:.3f}',9,'Helvetica-Bold')
    label(255,68,'Dice agreement between decoding rules',11,'Helvetica-Bold')
    label(32,41,'Size is foreground area divided by image area at native resolution using >127.',9,color=muted)
    label(32,26,'Dots: median. Boxes: middle 50%. Whiskers: 10th-90th percentiles. Source: author TN3K train/validation masks.',9,color=muted)
    c.showPage();c.save()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mask_folder',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    write(a.mask_folder,a.out)
