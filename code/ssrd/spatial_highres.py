"""The same spatial RD target at896, matched to the completed896RD control."""
import sys,json,time,math,hashlib,argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT,WORK,split_rows,log_event
from cache import Images
from network import EfficientRD,loss_fn
from encoder_sdpa import enable_sdpa
from stable_optimizer import BatchedStableAdamW
from spatial_target import corrupt
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from train import FixedStepBatches
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="3cad",choices=["3cad","mvtec_ad2"])
    ap.add_argument("--variant",default="spatial_highres",choices=["spatial_highres","whole_highres"])
    a=ap.parse_args();ds=a.dataset
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(6271)
    source=WORK/f"runs/{ds}_plain_stable_s17/model_10000.pt"
    run=WORK/f"runs/{ds}_{a.variant}_s17";run.mkdir(exist_ok=True)
    if (run/"COMPLETE").exists():return
    encoder=enable_sdpa(load_encoder("cuda"));model=EfficientRD().cuda().train()
    model.load_state_dict(torch.load(source,weights_only=False,map_location="cpu")["model"])
    opt=BatchedStableAdamW(model.parameters(),lr=.0002)
    config={"dataset":ds,"variant":a.variant,"size":896,"steps":2000,"batch":8,"seed":6271,
        "source_checkpoint":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "control":f"{ds}_highres_continue_s17; same source, input size, sample order, steps, batch, LR and encoder precision",
        "max_lr":.0002,"min_lr":.00002,"warmup":50,"encoder":"fp16SDPA","decoder":"bf16",
        "corrupt_fraction":.5,"corruption":"same normalized-geometry image corruption as spatial_target.py",
        "target":("clean teacher inside synthetic support; observed corrupted teacher outside; clean samples retain normalRD" if a.variant=="spatial_highres" else "clean teacher at all positions; identical corruptions"),
        "checkpoint_rule":"final2000update only","inference":"unchanged ordinaryRD cosine maps"}
    (run/"config.json").write_text(json.dumps(config,indent=2));start_step=0
    if (run/"last.pt").exists():
        ck=torch.load(run/"last.pt",weights_only=False,map_location="cpu");model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        start_step=ck["step"];torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    rows=split_rows(ds,"train")
    loader=DataLoader(Images(rows,896),batch_sampler=FixedStepBatches(len(rows),8,6271,start_step,2000),
        num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(6271))
    mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None];std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    start=time.time();losses=[]
    for step,images in enumerate(loader,start_step+1):
        images=images.cuda(non_blocking=True)
        with torch.no_grad():
            raw=(images[:4]*std+mean).clamp(0,1)
            altered,mask=corrupt(raw)
            with torch.autocast("cuda",dtype=torch.float16):
                clean=encode_groups(encoder,images).to(torch.float16).float()
                changed=encode_groups(encoder,(altered-mean)/std).to(torch.float16).float()
            inputs=clean.clone();inputs[:4]=changed;target=inputs.clone()
            support=F.adaptive_max_pool2d(mask.float(),(64,64)).flatten(2).transpose(1,2)[:,None]>0
            target[:4,:,5:]=torch.where(support,clean[:4,:,5:],changed[:,:,5:])
            if a.variant=="whole_highres":target=clean
        s=step-1
        lr=.0002*(s+1)/50 if s<50 else .00002+.00009*(1+math.cos(math.pi*(s-50)/1950))
        for g in opt.param_groups:g["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss=loss_fn(target,model(inputs),10000+s)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,foreach=True);opt.step()
        losses.append(loss.detach())
        if step%100==0:
            val=torch.stack(losses).mean()
            if not bool(torch.isfinite(val)):raise FloatingPointError("nonfinite")
            log_event(run/"training.jsonl",step=step,loss=float(val),seconds=time.time()-start,steps_sec=(step-start_step)/(time.time()-start),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses=[]
        if step%500==0:
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"config":config,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},run/"last.pt.tmp")
            (run/"last.pt.tmp").replace(run/"last.pt")
    torch.save({"model":model.state_dict(),"step":12000,"config":config},run/"model_12000.pt")
    (run/"COMPLETE").write_text("2000");log_event(run/"training.jsonl",event="complete",seconds=time.time()-start)
if __name__=="__main__":main()
