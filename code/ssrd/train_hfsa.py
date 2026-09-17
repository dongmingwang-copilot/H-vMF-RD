"""Official HFSA architecture under the common normal-only multicategory split."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT,WORK,CACHE,split_rows,log_event
from cache import Images
from hfsa_adapter import HFSA
sys.path.insert(0,str(ROOT/"code"))
from train import FixedStepBatches
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True)
    ap.add_argument("--seed",type=int,default=17);ap.add_argument("--size",type=int,default=448)
    ap.add_argument("--steps",type=int,default=12000);ap.add_argument("--batch",type=int,default=16)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    run=WORK/f"runs/{a.dataset}_hfsa{a.size}_s{a.seed}";run.mkdir(exist_ok=True)
    if (run/"COMPLETE").exists():return
    resume_dir=(CACHE/"hfsa_resume")/run.name;resume_dir.mkdir(parents=True,exist_ok=True)
    resume=resume_dir/"last.pt"
    model=HFSA(a.size).cuda().to(memory_format=torch.channels_last).train()
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=.001,betas=(.5,.999),fused=True)
    config={**vars(a),"lr":.001,"betas":[.5,.999],"precision":"bf16","optimizer":"Adamfused",
      "setting":"one model per dataset; all normal training categories",
      "checkpoint_rule":"final12000updates; no test selection",
      "training_images_seen":a.steps*a.batch,
      "upstream":json.loads((ROOT/"code/third_party/hfsa/SOURCE.json").read_text()),
      "adaptations":["common normal split","448input; axial relative embedding length28",
        "Haar transforms use standard autograd convolutions, verified against adjoint"],
      "parameters_trainable":sum(p.numel() for p in params)}
    (run/"config.json").write_text(json.dumps(config,indent=2))
    rows=split_rows(a.dataset,"train")
    (run/"split_ids.json").write_text(json.dumps({"train":[r["id"] for r in rows],"validation":[r["id"] for r in split_rows(a.dataset,"validation")]}))
    step=0
    if resume.exists():
        ck=torch.load(resume,map_location="cpu",weights_only=False)
        model.load_trainable(ck["model"]);opt.load_state_dict(ck["optimizer"]);step=ck["step"]
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);del ck
    loader=DataLoader(Images(rows,a.size),batch_sampler=FixedStepBatches(len(rows),a.batch,a.seed,step,a.steps),
      num_workers=6,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(a.seed))
    start=time.time();initial_step=step;losses=[]
    log_event(run/"training.jsonl",event="start",step=step,config=config)
    for images in loader:
        images=images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):teacher,student=model(images);loss=model.loss(teacher,student)
        loss.backward();opt.step();step+=1;losses.append(loss.detach())
        if step%100==0:
            values=torch.stack(losses)
            if not bool(torch.isfinite(values).all()):raise FloatingPointError("nonfinite HFSA loss")
            log_event(run/"training.jsonl",event="progress",step=step,loss=float(values.mean()),seconds=time.time()-start,
              steps_sec=(step-initial_step)/(time.time()-start),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses=[]
        if step%1000==0:
            ck={"model":model.trainable_state(),"optimizer":opt.state_dict(),"config":config,"step":step,
              "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
            torch.save(ck,str(resume)+".tmp");Path(str(resume)+".tmp").replace(resume)
    torch.save({"model":model.trainable_state(),"config":config,"step":step},run/"model_12000.pt")
    (run/"COMPLETE").write_text(str(step))
    log_event(run/"training.jsonl",event="complete",step=step,seconds=time.time()-start)
if __name__=="__main__":main()
