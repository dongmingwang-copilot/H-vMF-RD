"""Joint GPU inference, followed by the original native-resolution metric protocol."""
import argparse,sys,json,hashlib,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from common import ROOT,WORK,split_rows,log_event
from cache import Images
from network import EfficientRD,anomaly_map
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from evaluate import category_metrics
from fast_metrics import apply as accelerate_metrics
accelerate_metrics()
def predict(source,target,step,seed=17,tag="",variants=("normal_continue","whole_denoise","spatial_target")):
    rows=split_rows(target,"test")
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=True
    enc=load_encoder("cuda")
    models={};outputs={};maps={};scores={}
    for variant in variants:
        run=WORK/"runs"/f"{source}_{variant}{chr(95)+tag if tag else chr(0)[:0]}_s{seed}"
        output=run/f"eval_{target}_{step}";output.mkdir(exist_ok=True)
        checkpoint=run/f"model_{step}.pt"
        if (output/"PREDICTIONS_COMPLETE").exists():continue
        ck=torch.load(checkpoint,weights_only=False,map_location="cpu")
        model=EfficientRD().cuda().eval();model.load_state_dict(ck["model"])
        models[variant]=model;outputs[variant]=output;maps[variant]=[];scores[variant]=[]
        (output/"prediction_metadata.json").write_text(json.dumps({
            "checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "target":target,"source":source,"seed":seed,"step":step,"size":448,"gaussian_sigma":4.,
            "stored_maps":"raw token maps","image_score":"maximum of smoothed input-resolution map",
            "projection_fit_dataset":source if variant=="projected" else None,
            "manifest_sha256":hashlib.sha256((ROOT/"data"/target/"manifest.json").read_bytes()).hexdigest()},indent=2))
    if not models:return
    loader=DataLoader(Images(rows,448),batch_size=48,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)
    start=time.time();done=0
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16):teacher=encode_groups(enc,images.cuda(non_blocking=True))
            # Round to the same fp16 cache representation used during training.
            teacher=teacher.to(torch.float16).float()
            for variant,model in models.items():
                x=None
                with torch.autocast("cuda",dtype=torch.bfloat16):raw=anomaly_map(teacher,model(teacher,x))
                raw=raw.cpu().numpy()
                maps[variant].append(raw)
                for array in raw:
                    filtered=cv2.GaussianBlur(cv2.resize(array,(448,448),interpolation=cv2.INTER_LINEAR),(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT)
                    scores[variant].append(float(filtered.max()))
            done+=len(images)
            if done%1008<48:
                log_event(WORK/"evaluation.jsonl",event="inference",source=source,target=target,done=done,total=len(rows),images_sec=done/(time.time()-start))
    for variant,output in outputs.items():
        np.savez_compressed(output/"predictions.npz",maps=np.concatenate(maps[variant]).astype(np.float32),scores=np.asarray(scores[variant]))
        (output/"rows.json").write_text(json.dumps(rows))
        (output/"PREDICTIONS_COMPLETE").write_text(str(len(rows)))
    log_event(WORK/"evaluation.jsonl",event="inference_complete",source=source,target=target,step=step,seconds=time.time()-start)
def metrics(source,target,step,seed=17,workers=4,tag="",variants=("normal_continue","whole_denoise","spatial_target")):
    for variant in variants:
        output=WORK/"runs"/f"{source}_{variant}{chr(95)+tag if tag else chr(0)[:0]}_s{seed}"/f"eval_{target}_{step}"
        if (output/"metrics.json").exists():continue
        assert (output/"PREDICTIONS_COMPLETE").exists()
        rows=json.loads((output/"rows.json").read_text())
        categories=sorted({r["category"] for r in rows})
        tasks=[(str(output/"predictions.npz"),rows,[i for i,r in enumerate(rows) if r["category"]==category],
                category,.05 if target=="mvtec_ad2" else .3,262144) for category in categories]
        start=time.time();results=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(category_metrics,tasks):
                results.append(result);print(json.dumps(result),flush=True)
        summary={k:float(np.mean([r[k] for r in results])) for k in ["i_auroc","p_auroc","p_ap","aupro"]}
        report={"source":source,"target":target,"variant":variant,"step":step,"seed":seed,
                "mean":summary,"categories":results,"seconds":time.time()-start,"bins":262144,
                "aggregation":"category macro-average; native annotations; per-category region pooling"}
        (output/"metrics.json").write_text(json.dumps(report,indent=2))
        log_event(WORK/"evaluation.jsonl",event="metrics_complete",**{k:v for k,v in report.items() if k!="categories"})
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--target",required=True)
    ap.add_argument("--step",type=int,default=10000);ap.add_argument("--seed",type=int,default=17)
    ap.add_argument("--workers",type=int,default=4);ap.add_argument("--tag",default="");ap.add_argument("--variants",default="normal_continue,whole_denoise,spatial_target");ap.add_argument("--mode",choices=["predict","metrics","both"],default="both");a=ap.parse_args()
    if a.mode in {"predict","both"}:predict(a.source,a.target,a.step,a.seed,a.tag,a.variants.split(","))
    if a.mode in {"metrics","both"}:metrics(a.source,a.target,a.step,a.seed,workers=a.workers,tag=a.tag,variants=a.variants.split(","))
