"""Common448 RD/RD++ comparisons; upstream projection objective and noise."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from common import ROOT,WORK,CACHE,split_rows,log_event
from cnn_adapter import CNNAdapter,rdpp,simplex
sys.path.insert(0,str(ROOT/"code"))
from baselines import BaselineBatches

class Samples(Dataset):
    def __init__(self,dataset,variant,seed):
        self.rows=split_rows(dataset,"train");self.variant=variant;self.seed=seed
        self.folder=CACHE/f"{dataset}_rgb";self.rgb=None;self.noise=None
        meta=json.loads((self.folder/"metadata.json").read_text())
        assert meta["row_ids"]==[r["id"] for r in self.rows]
        self.shape=tuple(meta["shape"])
    def __len__(self):return len(self.rows)
    def __getitem__(self,item):
        index,step=item
        if self.rgb is None:self.rgb=np.memmap(self.folder/"rgb.dat",dtype=np.uint8,mode="r",shape=self.shape)
        image=np.array(self.rgb[index],dtype=np.float32)/255.
        mean=np.asarray([.485,.456,.406],np.float32);std=np.asarray([.229,.224,.225],np.float32)
        clean=torch.from_numpy(((image-mean)/std).transpose(2,0,1).copy())
        if self.variant=="rd":return clean,torch.empty(0)
        if self.noise is None:self.noise=simplex.Simplex_CLASS()
        rng=np.random.RandomState((self.seed*1000003+step*9176+index)%2**32)
        h,w=rng.randint(10,32,size=2);y,x=rng.randint(1,448-h),rng.randint(1,448-w)
        self.noise.newSeed(int(rng.randint(1,2**31-1)))
        noise=self.noise.rand_3d_octaves((3,h,w),6,.6).transpose(1,2,0)
        image[y:y+h,x:x+w]+=.2*noise
        changed=torch.from_numpy(((image-mean)/std).transpose(2,0,1).copy())
        return clean,changed

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--variant",choices=["rd","rdpp"],required=True)
    ap.add_argument("--steps",type=int,default=12000);ap.add_argument("--seed",type=int,default=17);ap.add_argument("--benchmark",action="store_true")
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    run=WORK/f"runs/{a.dataset}_{a.variant}448_s{a.seed}"
    if not a.benchmark and (run/"COMPLETE").exists():return
    if not a.benchmark:run.mkdir(exist_ok=True)
    model=CNNAdapter(448,a.variant).cuda().to(memory_format=torch.channels_last).train()
    groups=[{"params":list(model.bn.parameters())+list(model.decoder.parameters()),"lr":.005}]
    if a.variant=="rdpp":groups.append({"params":list(model.projection.parameters()),"lr":.001})
    opt=torch.optim.Adam(groups,betas=(.5,.999),fused=True)
    projection_loss=rdpp.Revisit_RDLoss() if a.variant=="rdpp" else None
    config={**vars(a),"size":448,"batch":16,"precision":"bf16","betas":[.5,.999],
      "lr_decoder":.005,"lr_projection":.001 if a.variant=="rdpp" else None,
      "noise":"official simplex6octaves,persistence0.6,amplitude0.2;h,w in[10,31]pixels",
      "setting":"one multicategory model per dataset; common normal split",
      "score":"sum of native three-level cosine residuals,448resize,sigma4",
      "checkpoint_rule":"final prescribed update"}
    resume_dir=(CACHE/"cnn_resume")/run.name
    if not a.benchmark:resume_dir.mkdir(parents=True,exist_ok=True)
    last=resume_dir/"last.pt";step=0
    if last.exists() and not a.benchmark:
        ck=torch.load(last,weights_only=False,map_location="cpu");step=ck["step"]
        model.load_trainable(ck["model"]);opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);del ck
    data=Samples(a.dataset,a.variant,a.seed)
    stop=8 if a.benchmark else a.steps
    loader=DataLoader(data,batch_sampler=BaselineBatches(len(data),16,a.seed,step,stop),
      num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=3,
      multiprocessing_context="spawn",generator=torch.Generator().manual_seed(a.seed))
    if not a.benchmark:
        (run/"config.json").write_text(json.dumps(config,indent=2))
        (run/"split_ids.json").write_text(json.dumps({"train":[r["id"] for r in data.rows]}))
    start=time.time();losses=[];times=[];initial=step
    for images,noisy in loader:
        images=images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        if a.benchmark:torch.cuda.synchronize();tic=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            t,s,p=model(images);loss=model.loss(t,s)
            if a.variant=="rdpp":
                with torch.no_grad():nt=model.encoder(noisy.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
                npj=model.projection(nt)
        if a.variant=="rdpp":
            loss=loss+.2*projection_loss([x.float().contiguous() for x in nt],
              [x.float().contiguous() for x in npj],[x.float().contiguous() for x in p])
        loss.backward();opt.step();step+=1;losses.append(loss.detach())
        if a.benchmark:
            torch.cuda.synchronize();times.append(time.perf_counter()-tic)
            assert bool(torch.isfinite(loss))
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        elif step%100==0:
            vals=torch.stack(losses)
            if not bool(torch.isfinite(vals).all()):raise FloatingPointError("nonfinite CNN baseline")
            log_event(run/"training.jsonl",step=step,loss=float(vals.mean()),seconds=time.time()-start,
                      steps_sec=(step-initial)/(time.time()-start),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses=[]
        if not a.benchmark and step%1000==0:
            ck={"model":model.trainable_state(),"optimizer":opt.state_dict(),"step":step,"config":config,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
            torch.save(ck,str(last)+".tmp");Path(str(last)+".tmp").replace(last)
    if a.benchmark:
        report={"variant":a.variant,"finite":True,"median_step_sec":float(np.median(times[3:])),
                "peak_mb":torch.cuda.max_memory_allocated()/2**20,"config":config}
        (WORK/f"{a.variant}448_benchmark.json").write_text(json.dumps(report,indent=2));print(report);return
    torch.save({"model":model.trainable_state(),"config":config,"step":step},run/"model_12000.pt")
    (run/"COMPLETE").write_text(str(step))
    log_event(run/"training.jsonl",event="complete",step=step,seconds=time.time()-start)
if __name__=="__main__":main()
