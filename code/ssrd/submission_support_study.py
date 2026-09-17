"""Frozen-model diagnostic of synthetic-support localization; no training or selection."""
from pathlib import Path
import sys,json,hashlib,time,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch,cv2
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from common import ROOT,WORK,split_rows
from spatial_target import corrupt
from network import EfficientRD,anomaly_map
sys.path.insert(0,str(ROOT/"code"))
from cache_features import load_encoder,encode_groups
from metrics import PixelMetrics
from fast_metrics import apply
apply()
OUT=WORK/"submission_case_study";OUT.mkdir(exist_ok=True)
V=["normal_continue","whole_denoise","spatial_target"];SEEDS=[17,29,43]
def measure(task):
    key,cat,kind,ids=task
    cv2.setNumThreads(1);torch.set_num_threads(1)
    data=np.load(OUT/"support_diagnostic.npz")
    maps=data[key][ids];masks=data["masks"][ids]
    acc=PixelMetrics(float(maps.min()),float(maps.max()))
    for raw,mask in zip(maps,masks):
        smoothed=cv2.GaussianBlur(cv2.resize(raw,(448,448)),(0,0),4.,borderType=cv2.BORDER_REFLECT)
        acc.add(smoothed,mask)
    return {"model":key,"category":cat,"kind":kind,"images":len(ids),**acc.summary(.3)}
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=True
    rows=split_rows("3cad","validation");cats=sorted({r["category"] for r in rows})
    chosen=[]
    for cat in cats:
        chosen+=sorted([r for r in rows if r["category"]==cat],
            key=lambda r:hashlib.sha256(("support-diagnostic-v1:"+r["id"]).encode()).hexdigest())[:32]
    assert len(chosen)==256
    protocol={"status":"fixed before inference","row_ids":[r["id"] for r in chosen],
        "selection":"32 minimum SHA256(support-diagnostic-v1:id) per validation category",
        "categories":cats,"corruption_seed":20260917,"batch":32,"size":448,
        "line_patch":"16 line and 16 patch perturbations per category; original training generator",
        "checkpoints":"final 12000 updates for seeds 17,29,43; no fitting or selection",
        "metrics":"category-pooled pixel AP and AUPRO(FPR<=0.3); synthetic support at 448; macro average over eight categories"}
    (OUT/"support_protocol.json").write_text(json.dumps(protocol,indent=2))
    start=time.time()
    if not (OUT/"support_diagnostic.npz").exists():
        enc=load_encoder("cuda");models={};hashes={}
        for v in V:
            for seed in SEEDS:
                key=f"{v}_s{seed}";path=WORK/f"runs/3cad_{v}_s{seed}/model_12000.pt"
                model=EfficientRD().cuda().eval()
                model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["model"])
                models[key]=model;hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        outputs={key:[] for key in models};masks=[]
        mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
        torch.manual_seed(20260917)
        with torch.inference_mode():
            for start_i in range(0,len(chosen),32):
                ims=torch.stack([TF.to_tensor(Image.open(r["image"]).convert("RGB").resize((448,448),Image.Resampling.BICUBIC)) for r in chosen[start_i:start_i+32]]).cuda()
                altered,mask=corrupt(ims)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    features=encode_groups(enc,(altered-mean)/std).to(torch.float16).float()
                    for key,model in models.items():
                        outputs[key].append(anomaly_map(features,model(features)).cpu().numpy())
                masks.append(mask[:,0].cpu().numpy())
                print(json.dumps({"event":"inference","done":start_i+32,"seconds":time.time()-start}),flush=True)
        arrays={k:np.concatenate(v) for k,v in outputs.items()};arrays["masks"]=np.concatenate(masks)
        np.savez_compressed(OUT/"support_diagnostic.npz",**arrays)
        protocol["checkpoint_hashes"]=hashes
        (OUT/"support_protocol.json").write_text(json.dumps(protocol,indent=2))
        del enc,models,features,ims,altered,mask
        torch.cuda.empty_cache()
    tasks=[]
    for v in V:
        for seed in SEEDS:
            for ci,cat in enumerate(cats):
                for kind,parity in [("line",0),("patch",1)]:
                    tasks.append((f"{v}_s{seed}",cat,kind,list(range(ci*32+parity,(ci+1)*32,2))))
    records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(measure,tasks)):
            records.append(r)
            if (i+1)%24==0:print(json.dumps({"event":"metrics","done":i+1,"total":len(tasks),"seconds":time.time()-start}),flush=True)
    summary={}
    for v in V:
        summary[v]={}
        for kind in ["line","patch"]:
            summary[v][kind]={}
            for metric in ["p_ap","aupro"]:
                values=[np.mean([r[metric] for r in records if r["model"]==f"{v}_s{s}" and r["kind"]==kind])*100 for s in SEEDS]
                summary[v][kind][metric]={"values":values,"mean":float(np.mean(values)),"sd":float(np.std(values,ddof=1))}
    report={"protocol":protocol,"records":records,"summary":summary,"seconds":time.time()-start}
    (OUT/"support_results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(summary),flush=True)
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
