"""Export the paired changes and the three examples selected in the study protocol."""
import csv
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

from pairs import ARCHIVE, BASE, HERE, OUT, NoduleUNet, file_hash, group_masks, load, pairing, predict


def export():
    rows = load(OUT/"per-pair.json")
    if file_hash(OUT/"per-pair.json") != load(HERE/"results.json")["per_pair_sha256"]:
        raise ValueError("Scored records changed")
    fields = ["pair","id_A","id_B","correspondence","review_note","native_reference_dice",
              "surface_change_A","surface_change_B","surface_outcome","reversal_strength",
              "region_change_A","region_change_B"]
    with (OUT/"paired-changes.csv").open("w",newline="") as f:
        writer = csv.writer(f); writer.writerow(fields)
        for r in rows:
            status = "apparent same target" if r["same_target"] else "flagged"
            note = " / ".join(r["flags"]+[r["visual_flag"]] if r["visual_flag"] else r["flags"])
            values = [r["pair"],*r["ids"],status,note,r["native_dice"]]
            if r["unscorable"]:
                writer.writerow(values + ["","","unscorable","","",""])
                continue
            ds=r["surface_delta"]
            writer.writerow(values + [*ds,pairing(ds),min(map(abs,ds)) if pairing(ds)=="opposite" else "",*r["region_delta"]])
    same = [r for r in rows if r["same_target"] and not r["unscorable"]]
    strength = lambda r:min(map(abs,r["surface_delta"]))
    opposite = max((r for r in same if pairing(r["surface_delta"])=="opposite"), key=strength)
    concordant = max((r for r in same if pairing(r["surface_delta"]).startswith("both")),key=strength)
    unclear = min((r for r in rows if not r["same_target"]),key=lambda r:r["native_dice"])
    selected = [("Strongest reversal",opposite),("Strongest agreement",concordant),("Unclear target extent",unclear)]
    model = NoduleUNet()
    model.load_state_dict(torch.load(BASE,map_location="cpu",weights_only=True)["model"])
    model.eval()
    torch.set_num_threads(2)
    canvas=Image.new("RGB",(1140,800),"white")
    pen=ImageDraw.Draw(canvas)
    ink="#152c3c"
    font=ImageFont.load_default(size=17)
    title=ImageFont.load_default(size=25)
    pen.text((16,10),"How two reference outlines score the same one-pixel shrinkage",font=title,fill=ink)
    pen.text((16,54),"Ultrasound",font=font,fill=ink)
    pen.text((268,54),"Reference A (cyan) and B (orange)",font=font,fill=ink)
    pen.text((578,54),"Model before and after shrinkage",font=font,fill=ink)
    pen.text((878,54),"Change in surface Dice",font=font,fill=ink)
    with zipfile.ZipFile(ARCHIVE) as z, torch.inference_mode():
        for index,(label,r) in enumerate(selected):
            im, native = group_masks(z,r["ids"])
            p = predict(model,im,224)[0,0].numpy()>=.5
            small = ndi.binary_erosion(p)
            grid = p.shape[::-1]
            masks = [np.asarray(Image.fromarray(m).resize(grid,Image.Resampling.NEAREST)) for m in native]
            y,x = np.where(masks[0]|masks[1])
            pad = max(8,round(.22*max(np.ptp(y)+1,np.ptp(x)+1)))
            bounds = (max(0,int(x.min()-pad)),max(0,int(y.min()-pad)),min(grid[0],int(x.max()+1+pad)),min(grid[1],int(y.max()+1+pad)))
            base=np.array(im.resize(grid,Image.Resampling.BILINEAR).convert("RGB"))
            variants=[im.convert("RGB")]
            for contours,colors in ((masks,([0,220,255],[255,153,38])),((p,small),([228,70,176],[130,248,110]))):
                rgb=base.copy()
                edges=[m ^ ndi.binary_erosion(m) for m in contours]
                for edge,color in zip(edges,colors): rgb[edge]=color
                rgb[edges[0]&edges[1]]=255
                variants.append(Image.fromarray(rgb).crop(bounds))
            top=84+index*218
            for pic,left,width in zip(variants,(14,268,578),(241,290,280)):
                scale=min(width/pic.width,180/pic.height)
                pic=pic.resize((max(1,round(pic.width*scale)),max(1,round(pic.height*scale))), Image.Resampling.NEAREST if left>14 else Image.Resampling.LANCZOS)
                canvas.paste(pic,(left+(width-pic.width)//2,top+(180-pic.height)//2))
            pen.text((16,top+183),f"{r['pair']}   {label}",font=font,fill=ink)
            for i,(name,delta) in enumerate(zip(("A","B"),r["surface_delta"])):
                pen.text((878,top+55+i*34),f"Reference {name}   {delta:+.2f} points",font=font,fill=ink)
    pen.text((16,751),"Center and right show the same zoom on the evaluation grid. Model contours before (magenta) and after (green).",font=font,fill=ink)
    pen.text((16,775),"White marks overlapping outlines. All scores use the full image.",font=font,fill=ink)
    canvas.save(OUT/"meeting-examples.png")


if __name__=="__main__":
    export()
