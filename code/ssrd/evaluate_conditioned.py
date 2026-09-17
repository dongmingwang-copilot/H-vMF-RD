"""Paired 896 inference and native-mask evaluation with fixed spatial blur."""
import sys,json,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from common import ROOT,WORK,split_rows,log_event
from cache import Images
from network import EfficientRD,anomaly_map
from encoder_sdpa import enable_sdpa
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from metrics import PixelMetrics,image_auroc
from fast_metrics import apply
apply()
def predict(ds,variants):
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=True
    rows=split_rows(ds,"test");encoder=enable_sdpa(load_encoder("cuda"));models={};outs={};parts={};scores={}
    for v in variants:
        out=WORK/f"runs/{ds}_{v}_s17/eval_{ds}_12000";out.mkdir(parents=True,exist_ok=True)
        if (out/"PREDICTIONS_COMPLETE").exists():continue
        checkpoint=out.parent/"model_12000.pt"
        model=EfficientRD().cuda().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=False,map_location="cpu")["model"])
        models[v]=model;outs[v]=out;parts[v]=[];scores[v]=[]
        (out/"prediction_metadata.json").write_text(json.dumps({"size":896,"gaussian_sigma":8.,
            "blur_rule":"constant normalized spatial scale:448sigma4->896sigma8","map_size":64,
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"encoder":"fp16SDPA",
            "image_score":"maximum of smoothed896map","variant":v,"step":12000},indent=2))
    if not models:return
    loader=DataLoader(Images(rows,896),batch_size=16,num_workers=6,pin_memory=True,persistent_workers=True)
    start=time.time();cursor=0
    with torch.inference_mode():
        for images in loader:
            image=images.cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16):
                clean=encode_groups(encoder,image).to(torch.float16).float()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                for v,model in models.items():
                    inp=clean
                    raw=anomaly_map(clean,model(inp)).cpu().numpy()
                    parts[v].append(raw)
                    for arr in raw:
                        m=cv2.GaussianBlur(cv2.resize(arr,(896,896)),(0,0),sigmaX=8.,sigmaY=8.,borderType=cv2.BORDER_REFLECT)
                        scores[v].append(float(m.max()))
            cursor+=len(image)
            if cursor%1000<16:log_event(WORK/"conditioned_real.jsonl",dataset=ds,event="inference",done=cursor,total=len(rows),images_sec=cursor/(time.time()-start))
    for v,out in outs.items():
        np.savez_compressed(out/"predictions.npz",maps=np.concatenate(parts[v]),scores=np.asarray(scores[v]))
        (out/"rows.json").write_text(json.dumps(rows));(out/"PREDICTIONS_COMPLETE").write_text(str(cursor))
    log_event(WORK/"conditioned_real.jsonl",dataset=ds,event="inference_complete",seconds=time.time()-start)
def category(payload):
    path,cat,limit=payload;cv2.setNumThreads(1)
    from pathlib import Path
    out=Path(path);rows=json.loads((out/"rows.json").read_text());meta=json.loads((out/"prediction_metadata.json").read_text())
    a=np.load(out/"predictions.npz");maps=a["maps"];scores=a["scores"]
    ids=[i for i,r in enumerate(rows) if r["category"]==cat]
    acc=PixelMetrics(min(float(maps[i].min()) for i in ids),max(float(maps[i].max()) for i in ids));labels=[]
    for i in ids:
        m=cv2.GaussianBlur(cv2.resize(maps[i],(meta["size"],meta["size"])),(0,0),sigmaX=meta["gaussian_sigma"],sigmaY=meta["gaussian_sigma"],borderType=cv2.BORDER_REFLECT)
        r=rows[i]
        mask=np.asarray(Image.open(r["mask"]).convert("L"))>0 if r["mask"] else np.zeros((r["native_height"],r["native_width"]),np.uint8)
        acc.add(m,mask);labels.append(r["label"])
    return {"category":cat,"i_auroc":image_auroc(labels,scores[ids]),**acc.summary(limit)}
def metrics(ds,variants):
    for v in variants:
        out=WORK/f"runs/{ds}_{v}_s17/eval_{ds}_12000"
        if (out/"metrics.json").exists():continue
        rows=json.loads((out/"rows.json").read_text());cats=sorted({r["category"] for r in rows});start=time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            results=[]
            for r in pool.map(category,[(str(out),c,.05 if ds=="mvtec_ad2" else .3) for c in cats]):
                results.append(r);print(json.dumps(r),flush=True)
        report={"dataset":ds,"variant":v,"mean":{k:float(np.mean([r[k] for r in results])) for k in ["i_auroc","p_auroc","p_ap","aupro"]},"categories":results,"seconds":time.time()-start}
        (out/"metrics.json").write_text(json.dumps(report,indent=2));print(json.dumps({"dataset":ds,"variant":v,"mean":report["mean"]}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--variants",default="highres_continue");ap.add_argument("--mode",choices=["both","predict","metrics"],default="both");a=ap.parse_args()
    if a.mode in {"both","predict"}:
        predict(a.dataset,a.variants.split(","))
        import gc
        gc.collect();torch.cuda.empty_cache()
    if a.mode in {"both","metrics"}:metrics(a.dataset,a.variants.split(","))
