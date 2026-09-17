"""Original convolutional RD modules used by the common-input baselines."""
from pathlib import Path
import importlib.util,sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent.parent
def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/'code/third_party'/relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result
resnet=module("rd4ad_resnet","rd4ad/resnet.py")
de_resnet=module("rd4ad_decoder","rd4ad/de_resnet.py")
rdpp=module("rdpp_utils","rdpp/utils/utils_train.py")
simplex=module("rdpp_noise","rdpp/dataset/noise.py")
class BaselineBatches:
    def __init__(self,count,batch,seed,start,stop):
        from train import FixedStepBatches
        self.batches = FixedStepBatches(count,batch,seed,start,stop)
        self.start = start

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for step,batch in enumerate(self.batches,self.start):
            yield [(index,step) for index in batch]

class CNNReverse(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        self.variant = variant
        self.encoder,self.bn = resnet.wide_resnet50_2(pretrained=False)
        if pretrained:
            weights = ROOT/'pretrained/wide_resnet50_2-95faca4d.pth'
            weights.parent.mkdir(parents=True,exist_ok=True)
            if not weights.exists():
                torch.hub.download_url_to_file('https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',str(weights))
            state = torch.load(weights,map_location='cpu',weights_only=True)
            # The RD teacher exposes only the first three ResNet stages.
            filtered = {k:v for k,v in state.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered,strict=True)
        self.encoder.eval().requires_grad_(False)
        self.decoder = de_resnet.de_wide_resnet50_2(pretrained=False)
        self.projection = rdpp.MultiProjectionLayer(base=64) if variant=='rdpp' else nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            teacher = self.encoder(images)
        projected = self.projection(teacher)
        reconstruction = self.decoder(self.bn(projected))
        return teacher,reconstruction,projected

    def anomaly_map(self, images):
        teacher,student,_ = self(images)
        maps = [1-F.cosine_similarity(a.float(),b.float(),dim=1) for a,b in zip(teacher,student)]
        return torch.stack([F.interpolate(m[:,None],size=(64,64),mode='bilinear',align_corners=True)[:,0] for m in maps]).mean(0)
