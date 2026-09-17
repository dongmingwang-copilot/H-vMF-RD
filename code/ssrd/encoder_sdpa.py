"""Frozen DINO encoder attention using PyTorch's scaled-dot-product kernel."""
from types import MethodType
from torch.nn import functional as F
def forward_sdpa(self,x,attn_bias=None):
    if attn_bias is not None:raise ValueError("nested attention bias is not supported")
    b,n,c=x.shape
    q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    out=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
    out=out.transpose(1,2).reshape(b,n,c)
    return self.proj_drop(self.proj(out)),None
def enable_sdpa(encoder):
    assert not encoder.training
    for block in encoder.blocks:block.attn.forward=MethodType(forward_sdpa,block.attn)
    return encoder
