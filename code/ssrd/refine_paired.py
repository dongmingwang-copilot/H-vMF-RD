"""Paired normal/whole/selective refinement from a frozen ordinaryRD run."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT,WORK,CACHE,log_event
from network import EfficientRD,loss_fn
from spatial_target import TrainingPairs,corrupt
from stable_optimizer import BatchedStableAdamW
sys.path.insert(0,str(ROOT/"code"))
from train import FixedStepBatches
from cache_features import load_encoder,encode_groups
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    seed=2718+1009*(a.seed-17);torch.manual_seed(seed);np.random.seed(seed)
    variants=["normal_continue","whole_denoise","spatial_target"]
    runs={v:WORK/f"runs/{a.dataset}_{v}_s{a.seed}" for v in variants}
    if all((r/"COMPLETE_2000").exists() for r in runs.values()):return
    source=WORK/f"runs/{a.dataset}_plain_stable_s{a.seed}/model_10000.pt"
    initial=torch.load(source,map_location="cpu",weights_only=False)["model"]
    models={};opts={}
    config={"dataset":a.dataset,"initial_seed":a.seed,"continuation_seed":seed,"source":str(source),
      "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"steps":2000,"batch":16,"size":448,
      "max_lr":.0002,"min_lr":.00002,"warmup":50,"corrupt_fraction":.5,
      "pairing":"same inputs, teacher features, minibatches and dropout RNG state for all three target rules",
      "checkpoint_rule":"final update only","optimizer":"same BatchedStableAdamW"}
    for v,run in runs.items():
        run.mkdir(exist_ok=True);(run/"config.json").write_text(json.dumps({**config,"variant":v},indent=2))
        model=EfficientRD().cuda().train();model.load_state_dict(initial);models[v]=model
        opts[v]=BatchedStableAdamW(model.parameters(),lr=.0002)
    encoder=load_encoder("cuda")
    resume_dir=CACHE/"paired_resume";resume_dir.mkdir(exist_ok=True)
    last=resume_dir/f"{a.dataset}_s{a.seed}.pt";step=0
    if last.exists():
        ck=torch.load(last,weights_only=False,map_location="cpu");step=ck["step"]
        for v in variants:models[v].load_state_dict(ck["models"][v]);opts[v].load_state_dict(ck["optimizers"][v])
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);del ck
    data=TrainingPairs(a.dataset)
    loader=DataLoader(data,batch_sampler=FixedStepBatches(len(data),16,seed,step,2000),
      num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(seed))
    mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    start=time.time();initial_step=step;losses={v:[] for v in variants}
    for clean,images in loader:
        clean=clean.cuda(non_blocking=True).float();images=images[:8].cuda(non_blocking=True)
        with torch.no_grad():
            altered,mask=corrupt(images)
            with torch.autocast("cuda",dtype=torch.bfloat16):changed=encode_groups(encoder,(altered-mean)/std)
            changed=changed.to(torch.float16).float()
            inputs=clean.clone();inputs[:8]=changed
            support=F.adaptive_max_pool2d(mask.float(),(32,32)).flatten(2).transpose(1,2)[:,None]>0
            selected=inputs.clone();selected[:8,:,5:]=torch.where(support,clean[:8,:,5:],changed[:,:,5:])
        rate=.0002*(step+1)/50 if step<50 else .00002+.00009*(1+math.cos(math.pi*(step-50)/1950))
        dropout_state=torch.cuda.get_rng_state()
        for v in variants:
            torch.cuda.set_rng_state(dropout_state);opt=opts[v]
            for group in opt.param_groups:group["lr"]=rate
            opt.zero_grad(set_to_none=True)
            teacher=selected if v=="spatial_target" else clean
            inp=clean if v=="normal_continue" else inputs
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=loss_fn(teacher,models[v](inp),10000+step)
            loss.backward();torch.nn.utils.clip_grad_norm_(models[v].parameters(),.1,foreach=True);opt.step();losses[v].append(loss.detach())
        step+=1
        if step%100==0:
            values={v:float(torch.stack(losses[v]).mean()) for v in variants}
            if not all(math.isfinite(x) for x in values.values()):raise FloatingPointError("nonfinite paired loss")
            log_event(WORK/"paired_refinement.jsonl",dataset=a.dataset,seed=a.seed,step=step,loss=values,
              steps_sec=(step-initial_step)/(time.time()-start),seconds=time.time()-start,gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses={v:[] for v in variants}
        if step%1000==0:
            ck={"step":step,"models":{v:m.state_dict() for v,m in models.items()},
              "optimizers":{v:o.state_dict() for v,o in opts.items()},"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
            torch.save(ck,str(last)+".tmp");Path(str(last)+".tmp").replace(last)
    for v,run in runs.items():
        torch.save({"model":models[v].state_dict(),"config":{**config,"variant":v},"step":12000},run/"model_12000.pt")
        (run/"COMPLETE_2000").write_text(str(step))
    log_event(WORK/"paired_refinement.jsonl",dataset=a.dataset,seed=a.seed,event="complete",seconds=time.time()-start)
if __name__=="__main__":main()
