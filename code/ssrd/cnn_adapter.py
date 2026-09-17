"""Original RD / RD++ modules with common multicategory data and input size."""
import sys
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT
sys.path.insert(0,str(ROOT/"code"))
from baselines import CNNReverse,rdpp,simplex

class CNNAdapter(CNNReverse):
    def __init__(self,size=448,variant="rd"):
        super().__init__(variant);self.size=size
    def trainable_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith("encoder.")}
    def load_trainable(self,state):
        incompatible=self.load_state_dict(state,strict=False)
        assert not incompatible.unexpected_keys
        assert all(k.startswith("encoder.") for k in incompatible.missing_keys)
    @staticmethod
    def loss(teacher,student):
        return sum((1-F.cosine_similarity(t.float().flatten(1),s.float().flatten(1),dim=1)).mean() for t,s in zip(teacher,student))
    @staticmethod
    def residuals(teacher,student):
        return [1-F.cosine_similarity(t.float(),s.float(),dim=1) for t,s in zip(teacher,student)]
    @staticmethod
    def score_map(residuals,size):
        return sum(F.interpolate(m[:,None],size=(size,size),mode="bilinear",align_corners=True)[:,0] for m in residuals)
