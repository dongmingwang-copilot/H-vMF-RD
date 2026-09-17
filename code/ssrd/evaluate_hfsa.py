"""Native-mask evaluation of HFSA; store the three raw residual resolutions."""
import argparse,json,hashlib,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np,torch
from PIL import Image
from torch.utils.data import DataLoader
from common import ROOT,WORK,split_rows,log_event
from cache import Images
from hfsa_adapter import HFSA
sys.path.insert(0,str(ROOT/"code"))
from metrics import PixelMetrics,image_auroc
from fast_metrics import apply
apply()
def predict(source,target,size=448):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;cv2.setNumThreads(1)
    run=WORK/f"runs/{source}_hfsa{size}_s17";out=run/f"eval_{target}_12000";out.mkdir(exist_ok=True)
    if (out/"PREDICTIONS_COMPLETE").exists():return
    ck=run/"model_12000.pt"
    model=HFSA(size).cuda().to(memory_format=torch.channels_last).eval()
    model.load_trainable(torch.load(ck,map_location="cpu",weights_only=False)["model"])
    rows=split_rows(target,"test")
    loader=DataLoader(Images(rows,size),batch_size=16,num_workers=6,pin_memory=True,persistent_workers=True)
    metadata={"source":source,"target":target,"variant":f"hfsa{size}","size":size,"gaussian_sigma":4.,
      "map":"sum of three interpolated cosine residuals; align_corners=True, thenGaussian",
      "score":"max of smoothed input-resolution map","checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),
      "native_masks":True,"precision":"bf16","storage":"lossless float32 raw multi-resolution residuals"}
    (out/"prediction_metadata.json").write_text(json.dumps(metadata,indent=2))
    parts=[[],[],[]];scores=[];done=0;start=time.time()
    with torch.inference_mode():
        for images in loader:
            images=images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t,s=model(images);residuals=model.residuals(t,s)
                full=model.score_map(residuals,size).cpu().numpy()
            for p,m in zip(parts,residuals):p.append(m.cpu().numpy())
            for m in full:scores.append(float(cv2.GaussianBlur(m,(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT).max()))
            done+=len(images)
            if done%1000<16:log_event(run/"evaluation.jsonl",event="prediction",target=target,done=done,total=len(rows),images_sec=done/(time.time()-start))
    np.savez_compressed(out/"predictions.npz",**{f"level{i}":np.concatenate(p) for i,p in enumerate(parts)},scores=np.asarray(scores))
    (out/"rows.json").write_text(json.dumps(rows));(out/"PREDICTIONS_COMPLETE").write_text(str(done))
def category(payload):
    path,cat,limit=payload;torch.set_num_threads(1);cv2.setNumThreads(1);out=Path(path)
    meta=json.loads((out/"prediction_metadata.json").read_text());rows=json.loads((out/"rows.json").read_text())
    arrays=np.load(out/"predictions.npz");levels=[arrays[f"level{i}"] for i in range(3)];scores=arrays["scores"]
    ids=[i for i,r in enumerate(rows) if r["category"]==cat]
    # Sum bounds are conservative over all interpolations and smoothing.
    lower=sum(float(m[ids].min()) for m in levels);upper=sum(float(m[ids].max()) for m in levels)
    acc=PixelMetrics(lower,upper);labels=[]
    for i in ids:
        residuals=[torch.from_numpy(m[i:i+1]) for m in levels]
        full=HFSA.score_map(residuals,meta["size"])[0].numpy()
        pred=cv2.GaussianBlur(full,(0,0),sigmaX=meta["gaussian_sigma"],sigmaY=meta["gaussian_sigma"],borderType=cv2.BORDER_REFLECT)
        row=rows[i];mask=np.asarray(Image.open(row["mask"]).convert("L"))>0 if row["mask"] else np.zeros((row["native_height"],row["native_width"]),np.uint8)
        acc.add(pred,mask);labels.append(row["label"])
    return {"category":cat,"images":len(ids),"i_auroc":image_auroc(labels,scores[ids]),**acc.summary(limit)}
def metrics(source,target,size=448):
    out=WORK/f"runs/{source}_hfsa{size}_s17/eval_{target}_12000"
    if (out/"metrics.json").exists():return
    rows=json.loads((out/"rows.json").read_text());cats=sorted({r["category"] for r in rows});start=time.time()
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(category,[(str(out),cat,.05 if target=="mvtec_ad2" else .3) for cat in cats]):
            results.append(res);print(json.dumps(res),flush=True)
    report={"source":source,"target":target,"variant":f"hfsa{size}","mean":{k:float(np.mean([x[k] for x in results])) for k in ["i_auroc","p_auroc","p_ap","aupro"]},
      "categories":results,"seconds":time.time()-start,"aggregation":"category macro-average; native masks"}
    (out/"metrics.json").write_text(json.dumps(report,indent=2));print(json.dumps(report["mean"]),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True);ap.add_argument("--target",required=True)
    ap.add_argument("--size",type=int,default=448);ap.add_argument("--mode",choices=["both","predict","metrics"],default="both");a=ap.parse_args()
    if a.mode in ["predict","both"]:predict(a.source,a.target,a.size)
    if a.mode in ["metrics","both"]:metrics(a.source,a.target,a.size)
