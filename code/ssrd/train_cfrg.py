"""CFRG 3CAD comparison: official architecture, synthesis, and loss; fixed common budget."""
import argparse,json,random,time,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch,cv2
from torch.utils.data import Dataset,DataLoader
from common import ROOT,WORK,CACHE,split_rows,log_event
from cfrg_adapter import build_model,trainable_state,load_trainable,objective,upstream_augmentation,verify_source
import sys
sys.path.insert(0,str(ROOT/"code"))
from baselines import BaselineBatches

class Samples(Dataset):
    def __init__(self,seed):
        self.rows=split_rows("3cad","train");self.seed=seed;self.rgb=None;self.aug=None
        folder=CACHE/"3cad_rgb";self.rgb_path=folder/"rgb.dat"
        meta=json.loads((folder/"metadata.json").read_text())
        assert meta["row_ids"]==[r["id"] for r in self.rows];self.shape=tuple(meta["shape"])
        self.textures=sorted((ROOT/"data/texture_source/dtd/images").glob("*/*.jpg"))
        assert len(self.textures)==5640,len(self.textures)
    def __len__(self):return len(self.rows)
    def __getitem__(self,item):
        index,step=item
        if self.rgb is None:
            self.rgb=np.memmap(self.rgb_path,dtype=np.uint8,mode="r",shape=self.shape)
            torch.set_num_threads(1);cv2.setNumThreads(1)
            self.aug=upstream_augmentation()
        seed=(self.seed*1000003+step*9176+index)%2**32
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.aug.rot.seed_(seed)
        rgb=np.array(self.rgb[index])
        with Image.open(self.textures[index%len(self.textures)]) as image:
            texture=np.asarray(image.convert("RGB").resize((448,448),Image.Resampling.BILINEAR))
        altered,mask=self.aug.perlin_noise(rgb,texture,aug_prob=1.)
        normal=rgb.astype(np.float32)/255.
        mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
        return (torch.from_numpy(((normal-mean)/std).transpose(2,0,1).copy()),
                torch.from_numpy(((altered-mean)/std).transpose(2,0,1).copy()),torch.from_numpy(mask.copy()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=12000)
    ap.add_argument("--seed",type=int,default=17);ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--benchmark",action="store_true");a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    commit=verify_source()
    run=WORK/f"runs/3cad_cfrg448_s{a.seed}"
    if not a.benchmark and (run/"COMPLETE").exists():return
    run.mkdir(exist_ok=True)
    model=build_model().cuda().to(memory_format=torch.channels_last).train();model.teacher.eval()
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(parameters,lr=.0005,betas=(.5,.999),fused=True)
    config={"variant":"cfrg448","dataset":"3cad","seed":a.seed,"steps":a.steps,"batch":16,"size":448,
      "precision":"bf16 network; float32 cosine and BCE losses","optimizer":"Adam (official training code)",
      "lr":.0005,"betas":[.5,.999],"milestones":[9600,10800],"lr_decay":.2,
      "upstream_commit":commit,"synthesis":"official DTD/Perlin generator, aug_prob=1, beta uniform[0,0.8)",
      "texture_source":"DTD 5640 images","checkpoint_rule":"final prescribed update; no test checkpoint selection",
      "protocol":"one model across all 8 categories; same normal train split,448 input,batch16,12000updates,native masks",
      "input_resize":"same bicubic448 RGB derivatives as matched baselines; DTD bilinear per upstream",
      "score":"sum of 3 cosine residuals plus sigmoid segmentation; bilinear align_corners=True; Gaussian sigma4",
      "trainable_parameters":sum(p.numel() for p in parameters)}
    resume=CACHE/"cfrg_resume";resume.mkdir(exist_ok=True);last=resume/"last.pt"
    step=0
    if last.exists() and not a.benchmark:
        ck=torch.load(last,map_location="cpu",weights_only=False);step=ck["step"]
        load_trainable(model,ck["model"]);opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);del ck
    data=Samples(a.seed);stop=12 if a.benchmark else a.steps
    loader=DataLoader(data,batch_sampler=BaselineBatches(len(data),16,a.seed,step,stop),
      num_workers=a.workers,pin_memory=True,persistent_workers=True,prefetch_factor=2,
      multiprocessing_context="spawn",generator=torch.Generator().manual_seed(a.seed))
    if not a.benchmark:
        (run/"config.json").write_text(json.dumps(config,indent=2))
        (run/"split_ids.json").write_text(json.dumps({"train":[r["id"] for r in data.rows]}))
    start=time.time();initial=step;recent=[];times=[];previous=time.perf_counter()
    for normal,altered,mask in loader:
        normal=normal.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        altered=altered.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        mask=mask.cuda(non_blocking=True)
        rate=.0005*(.2**int(step>=9600))*(.2**int(step>=10800))
        for group in opt.param_groups:group["lr"]=rate
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):outputs=model(normal,altered)
        loss,parts=objective(outputs,mask)
        if not bool(torch.isfinite(loss)):raise FloatingPointError(f"nonfinite CFRG loss at step{step}")
        loss.backward();opt.step();step+=1;recent.append(parts.detach())
        if a.benchmark:
            torch.cuda.synchronize();now=time.perf_counter();times.append(now-previous);previous=now
            assert all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
            previous=time.perf_counter()
        elif step%100==0:
            vals=torch.stack(recent).mean(0)
            log_event(run/"training.jsonl",step=step,loss=float(vals.sum()),components=vals.tolist(),
                seconds=time.time()-start,steps_sec=(step-initial)/(time.time()-start),
                gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20,lr=rate)
            recent=[]
        if not a.benchmark and step%1000==0:
            torch.save({"model":trainable_state(model),"optimizer":opt.state_dict(),"step":step,
              "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"config":config},str(last)+".tmp")
            Path(str(last)+".tmp").replace(last)
    if a.benchmark:
        report={"time":time.time(),"finite_loss_and_gradients":True,"median_step_seconds":float(np.median(times[4:])),
          "estimated_training_hours":float(np.median(times[4:])*12000/3600),
          "gpu_peak_mb":torch.cuda.max_memory_allocated()/2**20,"config":config}
        (WORK/"cfrg448_benchmark.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    torch.save({"model":trainable_state(model),"config":config,"step":step},run/"model_12000.pt")
    (run/"COMPLETE").write_text(str(step));log_event(run/"training.jsonl",event="complete",step=step,seconds=time.time()-start)
if __name__=="__main__":main()
