"""Visual check of display masking in the three previously selected TNUI examples."""
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage
import torch
from border_check import scan_bounds
from pilot import ROOT,HERE,load_pair,NoduleUNet,native_prediction,file_hash,measure


def panel():
    torch.set_num_threads(2)
    d=ROOT/'local/tnui-location'; src=json.loads((d/'sample.json').read_bytes())
    rec={r['name']:r for r in src['rows']}
    rows=json.loads((d/'per-image.json').read_bytes())
    scores={r['name']:r for r in json.loads((d/'border-per-image.json').read_bytes())}
    rows.sort(key=lambda r:r['scores']['oracle']['dice']-r['scores']['center']['dice'])
    cp=ROOT/'runs/source-fold0/gray/checkpoint.pt'
    assert file_hash(cp)==json.loads((HERE/'results.json').read_bytes())['checkpoint_sha256']
    model=NoduleUNet();ck=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);del ck;model.eval()
    sheet=Image.new('RGB',(744,848),'white');draw=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('Helvetica.ttc',16)
    except OSError:
        font=ImageFont.load_default()
    draw.text((12,8),'Effect of removing predictions in horizontal display borders',fill='#152f3b',font=font)
    for j,t in enumerate(('Ultrasound and retained rows','Full-frame prediction','After border filtering')):
        draw.text((12+244*j,38),t,fill='#152f3b',font=font)
    with torch.inference_mode():
        for i,q in enumerate((.1,.5,.9)):
            row=rows[round((len(rows)-1)*q)];name=row['name'];(im,gt),_=load_pair(d,rec[name])
            mask=native_prediction(model,im,(0,0,*im.size))
            lo,hi=scan_bounds(np.asarray(im));masked=mask.copy();masked[:lo]=False;masked[hi:]=False
            a=measure(masked,gt)['dice'];assert np.isclose(a,scores[name]['dice'])
            raw=im.convert('RGB')
            pen=ImageDraw.Draw(raw);pen.line((0,lo,im.width-1,lo),fill='yellow',width=4);pen.line((0,hi-1,im.width-1,hi-1),fill='yellow',width=4)
            pics=[raw]
            for pred in (mask,masked):
                image=np.array(im.convert('RGB'));truth=gt^ndimage.binary_erosion(gt,iterations=3)
                outline=pred^ndimage.binary_erosion(pred,iterations=3)
                image[truth]=[35,225,255];image[outline]=[255,153,46];image[truth&outline]=255
                pics.append(Image.fromarray(image))
            y=64+i*252
            for j,pic in enumerate(pics):
                sheet.paste(pic.resize((224,224),Image.Resampling.LANCZOS),(12+244*j,y))
            draw.text((12,y+225),name,fill='#152f3b',font=font)
            draw.text((256,y+225),f"Dice {row['scores']['whole']['dice']:.3f}",fill='#152f3b',font=font)
            draw.text((500,y+225),f"Dice {a:.3f}",fill='#152f3b',font=font)
    draw.text((12,822),'Cyan = reference contour, orange = prediction, white = contour overlap.',fill='#152f3b',font=font)
    sheet.save(d/'border-examples.png')

if __name__=='__main__':panel()
