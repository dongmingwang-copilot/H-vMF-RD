"""High-resolution SS-RD timing; same computation boundary as benchmark_inference.py."""
import argparse,gc,hashlib,json,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from common import ROOT,WORK
def measure(method,batch):
    assert method=="spatial_highres", "This entry point measures only the completed SS-RD 896 checkpoint"
    from common import split_rows
    from cache import Images
    from network import EfficientRD,anomaly_map
    from hfsa_adapter import HFSA
    sys.path.insert(0,str(ROOT/"code"))
    from cache_features import load_encoder,encode_groups
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    rows=sorted(split_rows("3cad","validation"),key=lambda r:hashlib.sha256(("latency:"+r["id"]).encode()).hexdigest())[:16]
    dataset=Images(rows,896)
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
                dense=arr if method in ["hfsa448","cfrg448"] else cv2.resize(arr,(896,896))
                smoothed=cv2.GaussianBlur(dense,(0,0),sigmaX=8.,sigmaY=8.,borderType=cv2.BORDER_REFLECT)
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
    ap=argparse.ArgumentParser(description="Measure the frozen SS-RD 896 checkpoint with the 448 benchmark timing boundary.")
    ap.add_argument("--batch",type=int,choices=[1,16],required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    result=measure("spatial_highres",a.batch)
    result.update(size=896,gaussian_sigma=8.,warmup_batches=50,measured_batches=100)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
