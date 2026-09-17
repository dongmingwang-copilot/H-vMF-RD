"""HFSA official networks adapted to a specified image size and common splits.

Upstream tensors and reconstruction objective are retained. Two compatibility
adjustments are explicit: axial relative embeddings use size/16 positions;
Haar wavelet operations use ordinary differentiable convolutions for native
mixed-precision autograd. Forward values and float64 gradients match upstream
exactly; its extra trailing None gradient is accepted by PyTorch.
"""
from pathlib import Path
import importlib.util,sys
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT
def wavelet_forward(x,module):
    return student_src.wavelet_transform(x,module.wt_filter)
def wavelet_inverse(x,module):
    return student_src.inverse_wavelet_transform(x,module.iwt_filter)
def load_module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/"code/third_party/hfsa"/file)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
teacher_src=load_module("hfsa_official_teacher","teacher.py")
student_src=load_module("hfsa_official_student","student.py")
class HFSA(nn.Module):
    def __init__(self,size=448):
        super().__init__()
        self.size=size
        self.encoder,self.csab=teacher_src.wide_resnet50_2(pretrained=False)
        self.encoder.load_state_dict(torch.load(ROOT/"pretrained/wide_resnet50_2-95faca4d.pth",map_location="cpu",weights_only=True),strict=True)
        self.encoder.requires_grad_(False).eval()
        if size!=256:
            self.csab.afl=teacher_src.AxialFeatureLearning(3072,3072,groups=1,kernel_size=size//16)
        self.decoder=student_src.de_wide_resnet50_2(pretrained=False)
        for m in self.decoder.modules():
            if isinstance(m,student_src.MultiFrequencyResponseModule):
                m.wt_function=partial(wavelet_forward,module=m)
                m.iwt_function=partial(wavelet_inverse,module=m)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def trainable_state(self):
        return {"csab":self.csab.state_dict(),"decoder":self.decoder.state_dict()}
    def load_trainable(self,state):
        self.csab.load_state_dict(state["csab"]);self.decoder.load_state_dict(state["decoder"])
    def forward(self,x):
        with torch.no_grad():teacher=self.encoder(x)
        student=self.decoder(self.csab(teacher))
        return teacher,student
    @staticmethod
    def loss(teacher,student):
        return sum((1-F.cosine_similarity(t.float().flatten(1),s.float().flatten(1),dim=1)).mean() for t,s in zip(teacher,student))
    @staticmethod
    def residuals(teacher,student):
        return [1-F.cosine_similarity(t.float(),s.float(),dim=1) for t,s in zip(teacher,student)]
    @staticmethod
    def score_map(residuals,size):
        return sum(F.interpolate(m[:,None],size=(size,size),mode="bilinear",align_corners=True)[:,0] for m in residuals)
