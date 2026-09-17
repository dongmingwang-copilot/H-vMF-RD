"""Prototype: constrain denoising discrepancy to the synthesized defect support."""
import sys,json,time,math,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from common import ROOT,WORK,CACHE,cache_data,split_rows,log_event
from network import EfficientRD,loss_fn
from stable_optimizer import BatchedStableAdamW
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from train import FixedStepBatches
class TrainingPairs(Dataset):
    def __init__(self,dataset,need_images=True):
        self.need_images=need_images
        self.rgb=None
        _,self.meta,self.features=cache_data(dataset)
        self.rows=split_rows(dataset,"train")
        assert [r["id"] for r in self.rows]==self.meta["row_ids"][:self.meta["train_count"]]
        folder=CACHE/f"{dataset}_rgb"
        if need_images and (folder/"COMPLETE").exists():
            rgbmeta=json.loads((folder/"metadata.json").read_text())
            assert rgbmeta["row_ids"]==[r["id"] for r in self.rows]
            self.rgb=np.memmap(folder/"rgb.dat",dtype=np.uint8,mode="r",shape=tuple(rgbmeta["shape"]))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if not self.need_images:
            image=torch.empty(0)
        elif self.rgb is not None:
            image=TF.to_tensor(np.array(self.rgb[i]))
        else:
            with Image.open(self.rows[i]["image"]) as im:
                image=TF.to_tensor(im.convert("RGB").resize((448,448),Image.Resampling.BICUBIC))
        return torch.from_numpy(np.array(self.features[i])),image
def corrupt(images):
    b,_,h,w=images.shape;device=images.device
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=device),torch.linspace(-1,1,w,device=device),indexing="ij")
    cx=torch.rand(b,1,1,device=device)*1.4-.7;cy=torch.rand(b,1,1,device=device)*1.4-.7
    angle=torch.rand(b,1,1,device=device)*math.pi*2
    xx=(x-cx)*angle.cos()+(y-cy)*angle.sin()
    yy=-(x-cx)*angle.sin()+(y-cy)*angle.cos()
    scratch=(torch.arange(b,device=device)%2==0)[:,None,None]
    length=torch.rand(b,1,1,device=device)*.15+.12
    width=torch.rand(b,1,1,device=device)*.012+.006
    line=(xx.abs()<length)&(yy.abs()<width)
    rx=torch.rand(b,1,1,device=device)*.09+.045;ry=torch.rand(b,1,1,device=device)*.09+.045
    boundary=F.interpolate(torch.rand(b,1,7,7,device=device),size=(h,w),mode="bicubic",align_corners=False)[:,0]
    blob=(xx/rx).square()+(yy/ry).square()<(0.8+.4*boundary)
    mask=torch.where(scratch,line,blob)[:,None].float()
    alpha=F.avg_pool2d(mask,3,1,1)
    strength=torch.rand(b,1,1,1,device=device)*.5+.5
    donor=images.roll(1,0).rot90(1,(-2,-1))
    light=(torch.rand(b,1,1,1,device=device)>.5).float()
    scratch_color=images.mean((-2,-1),keepdim=True)+(.35*(2*light-1))
    target=torch.where(scratch[:,None],scratch_color,donor).clamp(0,1)
    altered=images*(1-alpha*strength)+target*(alpha*strength)
    return altered,alpha>0
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--variant",choices=["spatial_target","normal_continue","whole_denoise"],default="spatial_target")
    a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(2718);np.random.seed(2718)
    run=WORK/"runs"/f"{a.dataset}_{a.variant}_s17";run.mkdir(exist_ok=True)
    if (run/"COMPLETE_2000").exists() and (run/"model_12000.pt").exists():return
    source=WORK/"runs"/f"{a.dataset}_plain_stable_s17"/"model_10000.pt"
    config={**vars(a),"source_checkpoint":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "batch":16,"seed":2718,"max_lr":.0002,"minimum_lr":.00002,"corrupt_fraction":.5,
            "teacher":"same frozen DINOv2","architecture":"same plain RD","input_size":448}
    (run/"config.json").write_text(json.dumps(config,indent=2))
    model=EfficientRD().cuda()
    model.load_state_dict(torch.load(source,map_location="cpu",weights_only=False)["model"])
    optimizer=BatchedStableAdamW(model.parameters(),lr=.0002)
    encoder=load_encoder("cuda") if a.variant!="normal_continue" else None
    data=TrainingPairs(a.dataset,need_images=a.variant!="normal_continue")
    step=0
    if (run/"last.pt").exists():
        ck=torch.load(run/"last.pt",weights_only=False,map_location="cuda")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);step=ck["step"]
        torch.set_rng_state(ck["torch_rng"].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in ck["cuda_rng"]])
    if step>=a.steps:return
    loader=DataLoader(data,batch_sampler=FixedStepBatches(len(data),16,2718,step,a.steps),
        num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(2718))
    mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    start=time.time();previous=start;previous_step=step;losses=[]
    for clean,images in loader:
        model.train();clean=clean.cuda(non_blocking=True).float()
        inputs=clean.clone();target=clean
        if a.variant!="normal_continue":
            images=images[:8].cuda(non_blocking=True)
            with torch.no_grad():
                altered,mask=corrupt(images)
                with torch.autocast("cuda",dtype=torch.bfloat16):encoded=encode_groups(encoder,(altered-mean)/std)
                encoded=encoded.to(torch.float16).float()
                inputs[:8]=encoded
                if a.variant=="spatial_target":
                    support=F.adaptive_max_pool2d(mask.float(),(32,32)).flatten(2).transpose(1,2)[:,None]>0
                    target=inputs.clone()
                    target[:8,:,5:]=torch.where(support,clean[:8,:,5:],encoded[:,:,5:])
        rate=.0002*(step+1)/50 if step<50 else .00002+.00009*(1+math.cos(math.pi*(step-50)/(a.steps-50)))
        for g in optimizer.param_groups:g["lr"]=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):decoded=model(inputs);loss=loss_fn(target,decoded,10000+step)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,foreach=True);optimizer.step()
        losses.append(loss.detach());step+=1
        if step%100==0 or step==a.steps:
            values=torch.stack(losses)
            if not bool(torch.isfinite(values).all()):raise FloatingPointError("nonfinite")
            now=time.time()
            log_event(run/"training.jsonl",step=step,event="progress",loss=float(values.mean()),gradient=float(grad),
                seconds=now-start,steps_sec=(step-previous_step)/(now-previous),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            previous=now;previous_step=step;losses=[]
        if step%1000==0 or step==a.steps:
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":config,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},run/"last.pt.tmp")
            (run/"last.pt.tmp").replace(run/"last.pt")
    torch.save({"model":model.state_dict(),"step":step,"config":config},run/"model_12000.pt")
    (run/"COMPLETE_2000").write_text(str(step))
    log_event(run/"training.jsonl",event="complete",step=step,seconds=time.time()-start)
if __name__=="__main__":main()
