"""Render the one-page supervisor handout from verified, aggregate development results."""
import json
from pathlib import Path
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

def render(root):
    read=lambda name: json.loads((root/'results'/name).read_bytes())
    fusion=read('fusion-audit.json')
    verified=read('stage-verification.json')
    methods=fusion['methods']
    output=root/'results/supervisor-meeting-summary.pdf'
    c=Canvas(str(output),pagesize=(612,792))
    c.setTitle('Thyroid nodule segmentation | MSc thesis direction meeting')
    def line(y,s,size=11,bold=False,x=38,color='#1B3440'):
        c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        c.setFillColor(HexColor(color));c.drawString(x,y,s)
    def rule(y):
        c.setStrokeColor(HexColor('#BCD0D0'));c.line(38,y,574,y)
    line(747,'Thyroid nodule segmentation on ultrasound',23,True)
    line(722,'Mehran Bakhtiari  |  MSc thesis discussion with Dr. Mina Amiri  |  21 November 2026',10)
    rule(708)
    line(676,'What the development study shows',15,True)
    line(652,f"{verified['training_images']:,} training and {verified['validation_images']} validation images from TN3K. The four matched ResNet34 U-Nets")
    line(635,'shared an initialization, fold, augmentation, training budget and evaluator.')
    line(606,'Input channels alone gave little average gain. Across the four fits, foreground Dice')
    line(589,'ranged from 0.8147 to 0.8167. Individual nodules showed gains and losses.')
    line(551,'Two ways to use a second prediction',14,True)
    for x,s in ((38,'Method'),(369,'Mean Dice'),(449,'Surface Dice*'),(536,'Misses')):
        line(525,s,9,True,x)
    for j,(name,label) in enumerate([('gray','Grayscale'),('contrast','Grayscale with local contrast'),('gray_contrast_average','Average of gray and contrast models'),('gray_flip_average','Grayscale with horizontal-flip averaging')]):
        y=501-j*25;m=methods[name]
        line(y,label,11,name=='gray_flip_average')
        line(y,f"{m['dice']:.4f}",11,False,369)
        line(y,f"{m['surface_dice']:.4f}",11,False,449)
        line(y,str(m['misses']),11,False,548)
    rule(412)
    line(394,'* Two-pixel tolerance on the 224-pixel long-side grid. Miss = zero reference overlap.',9)
    line(365,'Flip averaging gained 0.71 Dice points over single-pass grayscale (95% paired',11,True)
    line(348,'image interval +0.38 to +1.10). Both two-pass methods had seven complete misses.',11)
    line(324,'These are exploratory development results from one seed. Patient IDs are unavailable.',10)
    line(307,'DDTI external model performance is pending. The proposed cohort has 630 unique images.',10)
    rule(290)
    line(259,'The decision for this meeting',15,True)
    line(233,'1. Which correction matters most: finding the intended nodule, recovering missed')
    line(216,'    tissue, or avoiding adjacent tissue along a poorly defined margin?')
    line(188,'2. Are measurement calipers available and appropriate at the intended point of use?')
    line(160,'3. What improvement is worth pursuing, and who can judge ambiguous boundaries?')
    line(123,'Start with the blinded images before viewing scores. Then choose the primary endpoint,')
    line(106,'the closest published comparator, and an external cohort with an appropriate unit of analysis.')
    rule(85)
    line(63,'Sources: Gong et al., TN3K; Peng et al., TEM-Net; local matched development study and CPU replay.',9)
    line(46,'The four assigned architectures were reviewed. Direct same-protocol replication is a later comparison.',9)
    c.linkURL('https://doi.org/10.1016/j.compbiomed.2022.106389',(38,58,149,75))
    c.linkURL('https://doi.org/10.3390/bioengineering13070810',(149,58,253,75))
    c.showPage();c.save()


if __name__=='__main__':render(Path(__file__).resolve().parent)
