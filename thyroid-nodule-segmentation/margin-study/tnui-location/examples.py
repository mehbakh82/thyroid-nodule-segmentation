"""Render three prespecified quantiles of the location contrast with native-frame overlays."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
import torch

from pilot import ROOT,load_pair,localized_predictions,windows,measure,file_hash,NoduleUNet

def main():
    torch.set_num_threads(2)
    data=ROOT/'local/tnui-location';folder=Path(__file__).parent
    summary=json.loads((folder/'results.json').read_bytes())
    assert file_hash(data/'per-image.json')==summary['per_image_sha256']
    rows=json.loads((data/'per-image.json').read_bytes())
    rows.sort(key=lambda r:r['scores']['oracle']['dice']-r['scores']['center']['dice'])
    selection=[rows[round((len(rows)-1)*q)] for q in (.1,.5,.9)]
    sources={r['name']:r for r in json.loads((data/'sample.json').read_bytes())['rows']}
    cp=ROOT/'runs/source-fold0/gray/checkpoint.pt';assert file_hash(cp)==summary['checkpoint_sha256']
    net=NoduleUNet();ck=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);del ck;net.eval()
    tile=190;frames=('whole','center','oracle','automatic')
    panel=Image.new('RGB',(5*tile+20,760),'white');d=ImageDraw.Draw(panel)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14)
    d.text((10,8),'TNUI training subset   |   original anatomy, same frozen TN3K model',fill='black',font=font)
    d.text((10,32),'Cyan = reference, orange = prediction, white = agreement on contour, yellow = input crop.',fill='black',font=font)
    for k,title in enumerate(['Ultrasound','Full field','Central crop','Reference center','Automatic center']):
        d.text((10+k*tile,64),title,fill='black',font=font)
    with torch.inference_mode():
        for j,r in enumerate(selection):
            (im,gt),_=load_pair(data,sources[r['name']])
            preds,_,_=localized_predictions(net,im,gt)
            boxes,_=windows(gt,preds['whole'])
            y=87+j*219
            panel.paste(im.convert('RGB').resize((tile,tile)),(10,y))
            for k,name in enumerate(frames,1):
                mask=preds[name]
                assert np.isclose(measure(mask,gt)['dice'],r['scores'][name]['dice'])
                a=np.array(im.convert('RGB'));t=gt^ndi.binary_erosion(gt,iterations=3);p=mask^ndi.binary_erosion(mask,iterations=3)
                a[t]=[31,230,255];a[p]=[255,145,45];a[t&p]=255
                pic=Image.fromarray(a)
                if name in boxes:
                    l,top,right,b=boxes[name];ImageDraw.Draw(pic).rectangle((l,top,right-1,b-1),outline=(255,234,60),width=3)
                panel.paste(pic.resize((tile,tile)),(10+k*tile,y))
                d.text((10+k*tile,y+tile+1),f"Dice {r['scores'][name]['dice']:.3f}",fill='black',font=font)
            d.text((10,y+tile+1),r['name'],fill='black',font=font)
    d.text((10,746-10),'Rows: 10th, 50th and 90th percentile of reference-center minus central-crop Dice. Author data: zxg3017/TNUI-2021-.',fill='black',font=font)
    panel.save(data/'examples.png')

if __name__=='__main__':main()
