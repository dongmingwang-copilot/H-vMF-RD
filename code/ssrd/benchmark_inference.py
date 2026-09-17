"""Isolated-process inference timing with repeated measurement rounds."""
import argparse,gc,hashlib,json,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from common import ROOT,WORK
def measure(method,batch):
    from common import split_rows
    from cache import Images
    from network import EfficientRD,anomaly_map
    from hfsa_adapter import HFSA
    sys.path.insert(0,str(ROOT/"code"))
    from cache_features import load_encoder,encode_groups
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    rows=sorted(split_rows("3cad","validation"),key=lambda r:hashlib.sha256(("latency:"+r["id"]).encode()).hexdigest())[:16]
    dataset=Images(rows,448)
    images=torch.stack([dataset[i] for i in range(batch)]).cuda()
    checkpoint=WORK/f"runs/3cad_{method}_s17/model_12000.pt"
    if method=="cfrg448":
        from cfrg_adapter import build_model,load_trainable,score_map
        model=build_model().cuda().to(memory_format=torch.channels_last).eval()
        load_trainable(model,torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
        images=images.contiguous(memory_format=torch.channels_last)
    elif method=="hfsa448":
        model=HFSA(448).cuda().to(memory_format=torch.channels_last).eval()
        model.load_trainable(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
        images=images.contiguous(memory_format=torch.channels_last)
    else:
        encoder=load_encoder("cuda");model=EfficientRD().cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    elapsed=[]
    with torch.inference_mode():
        for i in range(150):
            if i==50:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if method=="cfrg448":
                    t,s,seg=model(images)
                    raw=score_map(t,s,seg,448).cpu().numpy()
                elif method=="hfsa448":
                    t,s=model(images)
                    raw=model.score_map(model.residuals(t,s),448).cpu().numpy()
                else:
                    t=encode_groups(encoder,images).to(torch.float16).float()
                    raw=anomaly_map(t,model(t)).cpu().numpy()
            for arr in raw:
                dense=arr if method in ["hfsa448","cfrg448"] else cv2.resize(arr,(448,448))
                smoothed=cv2.GaussianBlur(dense,(0,0),sigmaX=4.,sigmaY=4.,borderType=cv2.BORDER_REFLECT)
                score=float(smoothed.max())
            torch.cuda.synchronize();dt=time.perf_counter()-start
            if i>=50:elapsed.append(dt)
            if method in ["hfsa448","cfrg448"]:del s
            if method=="cfrg448":del seg
            del t,raw,dense,smoothed
    return {"method":method,"batch":batch,"seconds":elapsed,"peak_allocated_mb":torch.cuda.max_memory_allocated()/2**20,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--method");ap.add_argument("--batch",type=int);ap.add_argument("--output")
    ap.add_argument("--append-cfrg",action="store_true");a=ap.parse_args()
    if a.method:
        Path(a.output).write_text(json.dumps(measure(a.method,a.batch)));return
    folder=WORK/"inference_isolated";folder.mkdir(exist_ok=True)
    all_records={}
    methods=["cfrg448"] if a.append_cfrg else ["normal_continue","spatial_target","hfsa448"]
    order=[(m,b) for b in [1,16] for m in methods]
    for repeat in range(3):
        for method,batch in (order if repeat%2==0 else list(reversed(order))):
            output=folder/f"{method}_b{batch}_r{repeat}.json"
            subprocess.run([sys.executable,__file__,"--method",method,"--batch",str(batch),"--output",str(output)],check=True)
            rec=json.loads(output.read_text());all_records.setdefault((method,batch),[]).append(rec)
    records=[]
    for method in methods:
        for batch in [1,16]:
            rr=all_records[(method,batch)];seconds=np.asarray([v for x in rr for v in x["seconds"]])
            rec={"method":method,"batch":batch,"median_batch_ms":float(np.median(seconds)*1000),
                 "p95_batch_ms":float(np.quantile(seconds,.95)*1000),"images_sec":float(batch/seconds.mean()),
                 "peak_allocated_mb":max(x["peak_allocated_mb"] for x in rr),
                 "trainable_parameters":rr[0]["trainable_parameters"],"checkpoint_sha256":rr[0]["checkpoint_sha256"],
                 "round_medians_ms":[float(np.median(x["seconds"])*1000) for x in rr]}
            records.append(rec);print(json.dumps(rec),flush=True)
    report={"hardware":torch.cuda.get_device_name(),"size":448,"precision":"bf16; linear-attention accumulationfloat32",
            "timed":"encoder,decoder,map,device-to-host transfer,resize,Gaussian4andimage maximum",
            "not_timed":"image decoding, normalization, and input upload; normalized inputs resident on GPU","warmup_batches":50,"measured_batches_per_round":100,
            "rounds":3,"isolation":"fresh process per method,batch,round; alternating method order","records":records}
    if a.append_cfrg:
        old=json.loads((WORK/"inference_benchmark.json").read_text())
        assert old["rounds"]==3 and old["warmup_batches"]==50
        report["records"]=[x for x in old["records"] if x["method"]!="cfrg448"]+records
        report["cfrg_measurement"]="same isolated protocol after CFRG training; all jobs idle"
    (WORK/"inference_benchmark.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
