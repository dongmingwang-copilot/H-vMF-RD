"""Batched StableAdamW with device-side RMS update clipping."""
import math
import torch
from torch.optim import Optimizer
class BatchedStableAdamW(Optimizer):
    def __init__(self,params,lr=.002,betas=(.9,.999),eps=1e-10,weight_decay=1e-4,amsgrad=True):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,amsgrad=amsgrad))
    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            params=[p for p in group["params"] if p.grad is not None]
            if not params:continue
            grads=[p.grad for p in params]
            states=[]
            for p in params:
                state=self.state[p]
                if not state:
                    state.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p),
                                 max_exp_avg_sq=torch.zeros_like(p))
                state["step"]+=1;states.append(state)
            step=states[0]["step"]
            assert all(s["step"]==step for s in states)
            b1,b2=group["betas"]
            m=[s["exp_avg"] for s in states];v=[s["exp_avg_sq"] for s in states];vmax=[s["max_exp_avg_sq"] for s in states]
            torch._foreach_mul_(params,1-group["lr"]*group["weight_decay"])
            torch._foreach_mul_(m,b1);torch._foreach_add_(m,grads,alpha=1-b1)
            torch._foreach_mul_(v,b2);torch._foreach_addcmul_(v,grads,grads,value=1-b2)
            torch._foreach_maximum_(vmax,v)
            denom=torch._foreach_sqrt(vmax if group["amsgrad"] else v)
            torch._foreach_div_(denom,math.sqrt(1-b2**step));torch._foreach_add_(denom,group["eps"])
            scaled=torch._foreach_div(grads,denom)
            norms=torch.stack(torch._foreach_norm(scaled,2))
            sizes=norms.new_tensor([p.numel()**.5 for p in params])
            rates=(group["lr"]/(1-b1**step))/(norms/sizes).clamp_min(1)
            update=torch._foreach_div(m,denom)
            torch._foreach_mul_(update,list(rates.unbind()))
            torch._foreach_sub_(params,update)
