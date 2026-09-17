"""Bounded GPU compatibility and speed check before committing a full run."""
import json,time
import torch
from torch.utils.data import DataLoader
from common import WORK,split_rows
from cache import Images
from hfsa_adapter import HFSA,student_src
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
torch.manual_seed(17)
# Validate the ordinary autograd Haar path against its adjoint in float64.
filters,_=student_src.create_wavelet_filter("db1",3,3,torch.float64)
x=torch.randn(2,3,14,14,dtype=torch.float64,requires_grad=True)
y=student_src.wavelet_transform(x,filters);g=torch.randn_like(y)
(y*g).sum().backward()
expected=student_src.inverse_wavelet_transform(g,filters)
assert torch.allclose(x.grad,expected,rtol=1e-12,atol=1e-12)
batch=16
model=HFSA(448).cuda().to(memory_format=torch.channels_last).train()
images=next(iter(DataLoader(Images(split_rows("3cad","train")[:batch],448),batch_size=batch))).cuda().contiguous(memory_format=torch.channels_last)
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,betas=(.5,.999),fused=True)
times=[]
for step in range(8):
    torch.cuda.synchronize();start=time.time();opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        teacher,student=model(images);loss=model.loss(teacher,student)
    loss.backward()
    if step==0:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();torch.cuda.synchronize();times.append(time.time()-start)
report={"batch":batch,"size":448,"precision":"bf16","parameters_trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
  "parameters_all":sum(p.numel() for p in model.parameters()),"peak_gpu_mb":torch.cuda.max_memory_allocated()/2**20,
  "median_step_seconds":sorted(times[2:])[len(times[2:])//2],"loss":float(loss),"haar_adjoint_check":True}
(WORK/"hfsa_benchmark.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
