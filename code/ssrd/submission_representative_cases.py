"""Select explicitly performance-stratified illustrative gain and near-parity cases.

Category metrics choose two smaller positive and two smaller negative pooled
SS-RD/CFRG P-AP gaps. Within each positive category, the image nearest the 90th
percentile of all anomalous-image AP gaps illustrates a pronounced gain. Within
each negative category, both methods must have AP >= 50 and the image nearest
a -2-point negative gap illustrates near parity.
These images are selected for illustration after evaluation; they are not random
or typical samples. The full category table remains the performance comparison.
"""
from pathlib import Path
import os,json,hashlib,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2
from PIL import Image
from sklearn.metrics import average_precision_score
from common import ROOT,WORK
cv2.setNumThreads(1)
OUT=WORK/"submission_case_study/representative_cases"
OUT.mkdir(parents=True,exist_ok=True)
def load_metrics(variant,seed):
    return json.loads((WORK/f"runs/3cad_{variant}_s{seed}/eval_3cad_12000/metrics.json").read_text())
cfrg_metrics=load_metrics("cfrg448",17)
ssrd_metrics=[load_metrics("spatial_target",seed) for seed in [17,29,43]]
labels=["Camera cover","Tablet housing","Middle frame","New tablet housing","New middle frame","PC housing","Copper stator","Iron stator"]
table=[]
for cf,label in zip(cfrg_metrics["categories"],labels):
    cat=cf["category"]
    mean=np.mean([next(z["p_ap"] for z in m["categories"] if z["category"]==cat) for m in ssrd_metrics])*100
    table.append({"category":cat,"label":label,"p_ap":{"delta":float(mean-cf["p_ap"]*100)}})
pos=sorted([r for r in table if r["p_ap"]["delta"]>0],key=lambda r:abs(r["p_ap"]["delta"]))[:2]
neg=sorted([r for r in table if r["p_ap"]["delta"]<0],key=lambda r:abs(r["p_ap"]["delta"]))[:2]
cats=[r["category"] for r in sorted(pos,key=lambda r:table.index(r))+neg]
sp=WORK/"runs/3cad_spatial_target_s17/eval_3cad_12000"
cp=WORK/"runs/3cad_cfrg448_s17/eval_3cad_12000"
rows=json.loads((sp/"rows.json").read_text())
assert [r["id"] for r in rows]==[r["id"] for r in json.loads((cp/"rows.json").read_text())]
ss=np.load(sp/"predictions.npz")["maps"]
cf=np.load(cp/"maps.npy",mmap_mode="r")
def score(i):
    row=rows[i];gt=np.asarray(Image.open(row["mask"]).convert("L"))>0;h,w=gt.shape
    a=cv2.GaussianBlur(cv2.resize(ss[i],(448,448)),(0,0),4.,borderType=cv2.BORDER_REFLECT)
    a=cv2.resize(a,(w,h));b=cv2.resize(cf[i],(w,h))
    aa=float(average_precision_score(gt.ravel(),a.ravel()))*100
    bb=float(average_precision_score(gt.ravel(),b.ravel()))*100
    return {"id":row["id"],"category":row["category"],"index":i,"ssrd_ap":aa,"cfrg_ap":bb,"delta":aa-bb}
if __name__=="__main__":
    ids=[i for i,r in enumerate(rows) if r["category"] in cats and r["label"]]
    allrows=[];start=time.time()
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context("fork")) as ex:
        for k,z in enumerate(ex.map(score,ids,chunksize=8),1):
            allrows.append(z)
            if k%256==0:print(json.dumps({"completed":k,"total":len(ids),"seconds":round(time.time()-start,1)}),flush=True)
    (OUT/"per_image_ap.json").write_text(json.dumps(allrows))
    selected=[]
    for cat in cats:
        cr=[z for z in allrows if z["category"]==cat];median=float(np.median([z["delta"] for z in cr]))
        category_delta=next(r for r in table if r["category"]==cat)["p_ap"]["delta"]
        target=float(np.percentile([z["delta"] for z in cr],90)) if category_delta>0 else -2.0
        eligible=cr if category_delta>0 else [z for z in cr if z["ssrd_ap"]>=50 and z["cfrg_ap"]>=50 and z["delta"]<0]
        chosen=min(eligible,key=lambda z:(abs(z["delta"]-target),hashlib.sha256(("localization-audit:"+z["id"]).encode()).hexdigest()))
        summary=next(r for r in table if r["category"]==cat)
        selected.append({**chosen,"category_delta":summary["p_ap"]["delta"],"label":summary["label"],
         "anomalous_images":len(cr),"median_delta":median,"target_gap":target,"case_type":"pronounced gain" if category_delta>0 else "near parity",
         "gap_percentiles":np.percentile([z["delta"] for z in cr],[10,25,50,75,90]).tolist()})
    result={"category_rule":"two smallest positive and two smallest negative absolute category P-AP differences",
      "image_rule":"positive categories: nearest 90th percentile of all anomalous-image SS-RD minus CFRG AP gaps; negative categories: both AP >=50, negative gap nearest -2 percentage points; SHA256 identifier tie-break",
      "purpose":"performance-selected illustrative gain and near-parity cases; not random or typical examples and not an additional performance estimate",
      "seed":17,"per_image_grid":"native annotation","categories":cats,"entries":selected}
    (OUT/"selection.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
