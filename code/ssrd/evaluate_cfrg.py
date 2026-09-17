"""Complete CFRG predictions and common native-mask metrics; lossless float32 cloud RAM cache."""
import json,time,hashlib,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch,cv2
from PIL import Image
from torch.utils.data import DataLoader
from common import ROOT,WORK,CACHE,split_rows,log_event
from cache import Images
from cfrg_adapter import build_model,load_trainable,score_map
import sys
sys.path.insert(0,str(ROOT/"code"))
from metrics import PixelMetrics,image_auroc
from fast_metrics import apply
apply()
RUN=WORK/"runs/3cad_cfrg448_s17";OUT=RUN/"eval_3cad_12000"
def predict():
    OUT.mkdir(exist_ok=True)
    if (OUT/"PREDICTIONS_COMPLETE").exists():return
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    rows=split_rows("3cad","test");cache=CACHE/"cfrg_predictions";cache.mkdir(exist_ok=True)
    needed=len(rows)*448*448*4
    assert os.statvfs(cache).f_bavail*os.statvfs(cache).f_frsize>needed+2*2**30,"insufficient derived-cache capacity"
    ck=RUN/"model_12000.pt"
    model=build_model().cuda().to(memory_format=torch.channels_last).eval()
    load_trainable(model,torch.load(ck,map_location="cpu",weights_only=False)["model"])
    raw=cache/"maps.npy";maps=np.lib.format.open_memmap(raw,mode="w+",dtype=np.float32,shape=(len(rows),448,448))
    link=OUT/"maps.npy"
    if link.is_symlink():assert link.resolve()==raw;link.unlink()
    assert not link.exists();link.symlink_to(raw)
    scores=np.empty(len(rows),dtype=np.float64);start=time.time();done=0
    loader=DataLoader(Images(rows,448),batch_size=16,num_workers=4,pin_memory=True,persistent_workers=True)
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t,reconstructed,seg=model(images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            values=score_map(t,reconstructed,seg).cpu().numpy()
            for k,v in enumerate(values):
                smoothed=cv2.GaussianBlur(v,(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT)
                maps[done+k]=smoothed;scores[done+k]=float(smoothed.max())
            done+=len(values)
            if done%1000<16:log_event(RUN/"evaluation.jsonl",done=done,total=len(rows),images_sec=done/(time.time()-start))
    maps.flush();np.save(OUT/"scores.npy",scores);(OUT/"rows.json").write_text(json.dumps(rows))
    (OUT/"prediction_metadata.json").write_text(json.dumps({"source":"3cad","target":"3cad","variant":"cfrg448",
      "size":448,"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"native_masks":True,
      "storage":"float32 memorymapped448 smoothed maps; derived cloud RAM cache; persistent model and metrics",
      "map":"sum of cosine reconstruction residuals plus sigmoid segmentation, Gaussian sigma4",
      "score":"map maximum"},indent=2))
    (OUT/"PREDICTIONS_COMPLETE").write_text(str(done))
def category(cat):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    rows=json.loads((OUT/"rows.json").read_text());maps=np.load(OUT/"maps.npy",mmap_mode="r");scores=np.load(OUT/"scores.npy")
    ids=[i for i,r in enumerate(rows) if r["category"]==cat]
    low=min(float(maps[i].min()) for i in ids);high=max(float(maps[i].max()) for i in ids)
    acc=PixelMetrics(low,high);labels=[]
    for i in ids:
        row=rows[i]
        mask=np.asarray(Image.open(row["mask"]).convert("L"))>0 if row["mask"] else np.zeros((row["native_height"],row["native_width"]),np.uint8)
        acc.add(maps[i],mask);labels.append(row["label"])
    return {"category":cat,"images":len(ids),"i_auroc":image_auroc(labels,scores[ids]),**acc.summary(.3)}
def metrics():
    if (OUT/"metrics.json").exists():return
    rows=json.loads((OUT/"rows.json").read_text());cats=sorted({r["category"] for r in rows});start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(category,cats):results.append(result);print(json.dumps(result),flush=True)
    report={"source":"3cad","target":"3cad","variant":"cfrg448","mean":{k:float(np.mean([x[k] for x in results])) for k in ["i_auroc","p_auroc","p_ap","aupro"]},
      "categories":results,"seconds":time.time()-start,"aggregation":"category macro-average; original native masks"}
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2));print(json.dumps(report["mean"]),flush=True)
def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    predict();metrics()
if __name__=="__main__":main()
