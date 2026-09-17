"""Reverse-distillation backbone, target loss, and anomaly map."""
import sys,math
from pathlib import Path
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"third_party/dinomaly"))
from models.vision_transformer import Block,bMlp,LinearAttention2
class MatmulAttention(LinearAttention2):
    def forward(self,x,attn_mask=None):
        batch,tokens,dim=x.shape
        qkv=self.qkv(x).reshape(batch,tokens,3,self.num_heads,dim//self.num_heads).permute(2,0,3,1,4)
        with torch.autocast(device_type=x.device.type,enabled=False):
            q,k,v=qkv.float().unbind(0)
            q=torch.exp(q.clamp(max=0))+q.clamp(min=0)
            k=torch.exp(k.clamp(max=0))+k.clamp(min=0)
            kv=k.transpose(-2,-1)@v
            denominator=(q*k.sum(-2,keepdim=True)).sum(-1,keepdim=True).clamp_min(1e-8)
            result=(q@kv)/denominator
        result=result.transpose(1,2).reshape(batch,tokens,dim).to(qkv.dtype)
        return self.proj_drop(self.proj(result)),kv
class EfficientRD(nn.Module):
    def __init__(self):
        super().__init__()
        dimension=384
        self.bottleneck=bMlp(dimension,dimension*4,dimension,drop=.2)
        self.decoder=nn.ModuleList([
            Block(dim=dimension,num_heads=dimension//64,mlp_ratio=4.,qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm,eps=1e-8),attn=MatmulAttention)
            for _ in range(8)])
        self.apply(self.initialize)
    @staticmethod
    def initialize(module):
        if isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=.01,a=-.03,b=.03)
            if module.bias is not None:nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm):
            nn.init.ones_(module.weight);nn.init.zeros_(module.bias)
    def forward(self,teacher,bottleneck_input=None):
        x=self.bottleneck(teacher.mean(1) if bottleneck_input is None else bottleneck_input)
        first=[];second=[]
        for i,block in enumerate(self.decoder):
            x=block(x)
            (first if i<4 else second).append(x)
        return torch.stack([sum(second)/4,sum(first)/4],dim=1)
def loss_fn(teacher,decoded,step):
    target=teacher[:,:,5:].detach().float()
    student=decoded[:,:,5:].float()
    proportion=min(.9*step/1000,.9)
    with torch.no_grad():
        residual=1-F.cosine_similarity(target,student,dim=-1)
        flat=residual.transpose(0,1).flatten(1)
        count=max(1,int(flat.shape[-1]*(1-proportion)))
        threshold=torch.topk(flat,count,dim=-1,sorted=False).values.min(-1).values
        weight=torch.where(residual<threshold[None,:,None],.1,1.).unsqueeze(-1)
    student=student.detach()+(student-student.detach())*weight
    return (1-F.cosine_similarity(target.flatten(2),student.flatten(2),dim=-1)).mean()
def anomaly_map(teacher,decoded):
    score=(1-F.cosine_similarity(teacher[:,:,5:].float(),decoded[:,:,5:].float(),dim=-1)).mean(1)
    side=math.isqrt(score.shape[-1])
    return score.reshape(-1,side,side)
