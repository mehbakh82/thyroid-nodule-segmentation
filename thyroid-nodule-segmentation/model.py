"""A three-input ResNet34 U-Net. Both arms use identical parameter shapes."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34


class DecoderBlock(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(incoming+skip,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True),
            nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True))

    def forward(self,x,skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
        return self.layers(torch.cat([x,skip],1))


class NoduleUNet(nn.Module):
    def __init__(self,weights=None):
        super().__init__()
        enc=resnet34(weights=None)
        if weights is not None:
            enc.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        self.stem=nn.Sequential(enc.conv1,enc.bn1,enc.relu)
        self.level1=nn.Sequential(enc.maxpool,enc.layer1)
        self.level2,self.level3,self.level4=enc.layer2,enc.layer3,enc.layer4
        self.decoder=nn.ModuleList([DecoderBlock(512,256,128),DecoderBlock(128,128,64),
                                   DecoderBlock(64,64,32),DecoderBlock(32,64,16)])
        self.logits=nn.Conv2d(16,1,1)

    def forward(self,x):
        x0=self.stem(x)
        x1=self.level1(x0)
        x2=self.level2(x1)
        x3=self.level3(x2)
        z=self.level4(x3)
        for block,skip in zip(self.decoder,(x3,x2,x1,x0)):
            z=block(z,skip)
        return F.interpolate(self.logits(z),size=x.shape[-2:],mode='bilinear',align_corners=False)
