"""896-input normal-continuation control."""
import sys,time,json,argparse,math,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT,WORK,split_rows,log_event
from cache import Images
from network import EfficientRD,loss_fn
from stable_optimizer import BatchedStableAdamW
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from encoder_sdpa import enable_sdpa
from train import FixedStepBatches
def main(ds):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(6271)
    source=WORK/f"runs/{ds}_plain_stable_s17/model_10000.pt"
    initial=torch.load(source,weights_only=False,map_location="cpu")["model"]
    encoder=enable_sdpa(load_encoder("cuda"))
    variants=["highres_continue"]
    models={};optimizers={};runs={}
    config={"dataset":ds,"input_size":896,"steps":2000,"batch":8,"seed":6271,"max_lr":.0002,"min_lr":.00002,
        "source_checkpoint":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "target":"unfiltered frozen teacher groups; same cosine RD objective",
        "encoder_precision":"fp16 SDPA; decoder bf16",
        "paired_dropout":"same CUDA RNG state restored before each model forward","checkpoint_rule":"final update only"}
    for variant in variants:
        run=WORK/"runs"/f"{ds}_{variant}_s17";run.mkdir(exist_ok=True);runs[variant]=run
        (run/"config.json").write_text(json.dumps({**config,"variant":variant},indent=2))
        model=EfficientRD().cuda().train();model.load_state_dict(initial)
        models[variant]=model;optimizers[variant]=BatchedStableAdamW(model.parameters(),lr=.0002)
    start_step=0
    if all((runs[v]/"COMPLETE").exists() for v in variants):return
    if all((runs[v]/"last.pt").exists() for v in variants):
        points=[torch.load(runs[v]/"last.pt",weights_only=False,map_location="cpu") for v in variants]
        start_step=points[0]["step"]
        for v,ck in zip(variants,points):
            models[v].load_state_dict(ck["model"]);optimizers[v].load_state_dict(ck["optimizer"])
        torch.set_rng_state(points[0]["torch_rng"]);torch.cuda.set_rng_state_all(points[0]["cuda_rng"])
    rows=split_rows(ds,"train")
    loader=DataLoader(Images(rows,896),batch_sampler=FixedStepBatches(len(rows),8,6271,start_step,2000),
        num_workers=6,pin_memory=True,persistent_workers=True,prefetch_factor=3,generator=torch.Generator().manual_seed(6271))
    start=time.time();losses={v:[] for v in variants}
    for step,images in enumerate(loader,start_step+1):
        images=images.cuda(non_blocking=True)
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
            clean=encode_groups(encoder,images).to(torch.float16).float()
        state=torch.cuda.get_rng_state()
        for variant in variants:
            torch.cuda.set_rng_state(state)
            model=models[variant];opt=optimizers[variant]
            s=step-1
            lr=.0002*(s+1)/50 if s<50 else .00002+.00009*(1+math.cos(math.pi*(s-50)/1950))
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
            inputs=clean
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=loss_fn(clean,model(inputs),10000+s)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1,foreach=True);opt.step()
            losses[variant].append(loss.detach())
        if step%100==0:
            stats={v:float(torch.stack(losses[v]).mean()) for v in variants}
            if not all(math.isfinite(x) for x in stats.values()):raise FloatingPointError("nonfinite")
            losses={v:[] for v in variants}
            log_event(WORK/"conditioned_training.jsonl",dataset=ds,step=step,loss=stats,seconds=time.time()-start,steps_sec=(step-start_step)/(time.time()-start),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
        if step%500==0:
            shared={"step":step,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"config":config}
            for variant in variants:
                torch.save({**shared,"model":models[variant].state_dict(),"optimizer":optimizers[variant].state_dict()},runs[variant]/"last.pt.tmp")
            for variant in variants:(runs[variant]/"last.pt.tmp").replace(runs[variant]/"last.pt")
    for variant in variants:
        torch.save({"model":models[variant].state_dict(),"step":12000,"config":{**config,"variant":variant}},runs[variant]/"model_12000.pt")
        (runs[variant]/"COMPLETE").write_text("2000")
    log_event(WORK/"conditioned_training.jsonl",dataset=ds,event="complete",seconds=time.time()-start)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);a=ap.parse_args();main(a.dataset)
