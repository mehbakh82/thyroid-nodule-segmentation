"""Export the two prespecified examples for the supervisor meeting."""
import csv
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from pairs import ARCHIVE, HERE, OUT, file_hash, group_masks, load


def overlay(image, masks, colors, crop):
    rgb=np.array(image.convert("RGB"))
    edges=[m ^ ndi.binary_erosion(m) for m in masks]
    for edge,color in zip(edges,colors,strict=True):
        rgb[edge]=color
    rgb[edges[0]&edges[1]]=255
    return Image.fromarray(rgb).crop(crop)


def render():
    source = OUT/"model-ranking.csv"
    mp = OUT/"model-ranking-predictions.npz"
    receipt = load(HERE/"model-ranking-results.json")
    if file_hash(source)!=receipt["scores_sha256"] or file_hash(mp)!=receipt["predictions_sha256"]:
        raise ValueError("Score or prediction receipt mismatch")
    with source.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["same_target"]=="True"]
    def changes(r):
        return [float(r["surface2_delta_"+i]) for i in "AB"]
    cases = []
    for sign,title in ((-1,"Opposite winners"),(1,"Same winner")):
        eligible=[r for r in rows if sign*np.prod(changes(r))>0]
        if eligible:
            cases.append((title,max(eligible,key=lambda r:min(map(abs,changes(r))))))
    page=Image.new("RGB",(1080,140+225*len(cases)),"white")
    draw=ImageDraw.Draw(page)
    ink="#192e40"
    titlefont=ImageFont.load_default(size=24)
    font=ImageFont.load_default(size=16)
    draw.text((14,9),"Thyroid nodule boundaries scored against two outlines",font=titlefont,fill=ink)
    for label,x in (("Full scan",14),("A in cyan, B in orange",250),("Grayscale in purple, three-channel in green",492),("Surface-Dice difference",845)):
        draw.text((x,54),label,font=font,fill=ink)
    with zipfile.ZipFile(ARCHIVE) as z,np.load(mp) as saved:
        for i,(title,r) in enumerate(cases):
            im, raw=group_masks(z,[r["id_A"],r["id_B"]])
            size=int(r["width"]),int(r["height"])
            refs=[np.asarray(Image.fromarray(m).resize(size,Image.Resampling.NEAREST)) for m in raw]
            preds=[np.unpackbits(saved[r["pair"]+"_"+arm],axis=1)[:,:size[0]].astype(bool) for arm in ("gray","edge")]
            yy,xx=np.where(refs[0]|refs[1]|preds[0]|preds[1])
            pad=max(5,round(.15*max(np.ptp(yy)+1,np.ptp(xx)+1)))
            crop=(max(0,int(xx.min()-pad)),max(0,int(yy.min()-pad)),min(size[0],int(xx.max()+pad+1)),min(size[1],int(yy.max()+pad+1)))
            base=im.resize(size,Image.Resampling.BILINEAR)
            pictures=[im,overlay(base,refs,([25,215,246],[255,149,47]),crop),
                      overlay(base,preds,([229,103,226],[137,243,109]),crop)]
            top=84+225*i
            for p,x,width in zip(pictures,(13,250,518),(224,236,272),strict=True):
                scale=min(width/p.width,164/p.height)
                displayed=p.resize(tuple(max(1,round(n*scale)) for n in p.size),
                                   Image.Resampling.LANCZOS if x==13 else Image.Resampling.NEAREST)
                page.paste(displayed,(x+(width-displayed.width)//2,top+(164-displayed.height)//2))
            draw.text((14,top+174),r["pair"]+"   "+title,fill=ink,font=font)
            for j,(ref,delta) in enumerate(zip("AB",changes(r),strict=True)):
                draw.text((845,top+44+34*j),f"Using {ref}   {delta:+.2f} points",fill=ink,font=font)
    draw.text((14,page.height-53),"Positive values favor the three-channel model. White marks coincident outlines within a panel.",fill=ink,font=font)
    draw.text((14,page.height-30),"Each pair of contour panels shows the same field. Scores use the full unpadded image.",fill=ink,font=font)
    page.save(OUT/"model-ranking-examples.png")


if __name__=="__main__":
    render()
