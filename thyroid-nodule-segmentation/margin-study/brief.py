"""Generate a compact discussion page from the pilot's saved aggregate records."""
import json
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor

p=Path(__file__).resolve().parent
read=lambda f:json.loads((p/f).read_bytes())
r=read('results.json');oc=read('offset-check.json');head=read('opportunity.json')
c=Canvas(str(p/'meeting-update.pdf'),pagesize=(612,738))
c.setTitle('Thyroid nodule margin correction | Development pilot')
def put(x,y,t,size=11,bold=False,color='#183944'):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
    c.setFillColor(HexColor(color));c.drawString(x,y,t)
def rule(y):
    c.setStrokeColor(HexColor('#ADC1C5'));c.line(36,y,576,y)
put(36,700,'Can we correct a misplaced nodule margin?',22,True)
put(36,675,'Mehran Bakhtiari   |   MSc thesis discussion with Dr. Mina Amiri   |   23 November 2026',10)
rule(660)
put(36,632,'Working assumption',13,True)
put(36,611,'The nodule is already localized. An edit is valuable if it recovers a misplaced margin')
put(36,594,'while preserving tissue already traced correctly. An inference-time edit uses only the scan')
put(36,577,'and the baseline prediction. Two JPEG decoding conventions provide the training targets.')
put(36,539,'What happened in the pilot',15,True)
put(36,516,f"Five image folds, {r['images']} validation images. Primary comparison on {r['localized']} baseline-localized images.")
for x,t in [(36,'Contour method'),(342,'Primary reference'),(464,'Strict reference')]:put(x,486,t,10,True)
methods=[('baseline','Grayscale + flip baseline'),('active_contour','Geodesic active contour'),('single_head','Local classifier'),('dual_mean','Two classifiers, mean'),('selective','Two classifiers, strong agreement')]
for j,(method,label) in enumerate(methods):
    y=463-j*21;put(36,y,label,11,method=='selective')
    for x,k in [(351,'above127'),(480,'only255')]:put(x,y,f"{r['methods'][method][k]['localized_mean_surface2']:.5f}")
rule(361)
put(36,342,'Mean surface Dice at two pixels. Primary mask uses >127; strict mask uses exactly 255.',9)
put(36,314,'Primary gain: +0.009 points (95% image interval -0.019 to +0.043).',14,True)
put(36,291,'Among 534 localized images, the selective rule edited 35. On pixels where the labels')
put(36,274,'agreed, it corrected 340 errors and damaged 321 previously correct pixels.')
put(36,235,'A change of reference can reverse the verdict',15,True)
put(36,212,'Shrinking the same baseline predictions by one pixel changed surface Dice by')
put(36,190,f"{oc['above127']['delta_surface_dice_points']['shrink_one']:+.2f} points against the primary mask and {oc['only255']['delta_surface_dice_points']['shrink_one']:+.2f} against the strict mask.",13,True)
put(36,168,'This post-hoc control tests sensitivity to label encoding. The clinical boundary requires')
put(36,151,'independent evidence. The local correction missed the provisional two-point target.')
rule(127)
put(36,102,'Discussion: On which visible arcs would an expert request a correction?',13,True)
put(36,81,'Could we recover lossless contours or obtain a second blinded drawing for those cases?',11)
put(36,42,'Exploratory TN3K development analysis, one source-model seed, image-level grouping. External testing pending.',9)
c.showPage();c.save()
