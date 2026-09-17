import sys,os,json,time,math,argparse,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common import ROOT,WORK,cache_data,log_event
from network import EfficientRD,loss_fn
from stable_optimizer import BatchedStableAdamW
sys.path.insert(0,str(ROOT/"code"))
from train import FixedStepBatches
class FeaturePairs(Dataset):
    def __init__(self,dataset,projected,split="train"):
        cache,self.meta,self.features=cache_data(dataset)
        self.start=0 if split=="train" else self.meta["train_count"]
        self.count=self.meta["train_count"] if split=="train" else self.meta["validation_count"]
        self.projected=None
        if projected:
            assert (cache/"PROJECTED_COMPLETE").exists()
            self.projected=np.memmap(cache/"projected.dat",mode="r",dtype=np.float16,
                shape=(len(self.features),self.features.shape[2],self.features.shape[3]))
    def __len__(self):return self.count
    def __getitem__(self,i):
        i+=self.start
        y=torch.from_numpy(np.array(self.features[i]))
        z=torch.from_numpy(np.array(self.projected[i])) if self.projected is not None else torch.empty(0)
        return y,z
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True);ap.add_argument("--variant",choices=["plain"],required=True)
    ap.add_argument("--seed",type=int,default=17);ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--steps",type=int,default=10000);ap.add_argument("--stop",type=int,default=2500)
    ap.add_argument("--compile",action="store_true")
    ap.add_argument("--resume-root",default=None)
    ap.add_argument("--optimizer",choices=["fused","stable"],default="fused");ap.add_argument("--tag",default="")
    a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    run=WORK/"runs"/f"{a.dataset}_{a.variant}{chr(95)+a.tag if a.tag else chr(0)[:0]}_s{a.seed}"
    run.mkdir(parents=True,exist_ok=True)
    resume_dir=(Path(a.resume_root)/run.name) if a.resume_root else run
    resume_dir.mkdir(parents=True,exist_ok=True)
    args=vars(a)
    if (run/"config.json").exists():
        prior=json.loads((run/"config.json").read_text())
        assert all(prior[k]==args[k] for k in ("dataset","variant","seed","batch","steps","compile"))
    else:(run/"config.json").write_text(json.dumps(args,indent=2))
    train=FeaturePairs(a.dataset,a.variant=="projected")
    (run/"split_ids.json").write_text(json.dumps({"train":train.meta["row_ids"][:len(train)],"validation":train.meta["row_ids"][len(train):]}))
    model=EfficientRD().cuda()
    optimizer=(BatchedStableAdamW(model.parameters()) if a.optimizer=="stable" else torch.optim.AdamW(model.parameters(),lr=.002,betas=(.9,.999),weight_decay=1e-4,amsgrad=True,eps=1e-10,fused=True))
    step=0
    if (resume_dir/"last.pt").exists():
        ck=torch.load(resume_dir/"last.pt",map_location="cuda",weights_only=False)
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);step=ck["step"]
        torch.set_rng_state(ck["torch_rng"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        del ck
    if step>=a.stop:return
    active=torch.compile(model) if a.compile else model
    loader=DataLoader(train,batch_sampler=FixedStepBatches(len(train),a.batch,a.seed,step,a.stop),
        num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=3,
        generator=torch.Generator().manual_seed(a.seed))
    timer=time.time();last_time=timer;last_step=step;losses=[]
    log_event(run/"training.jsonl",event="start",step=step,stop=a.stop,parameters=sum(p.numel() for p in model.parameters()))
    for teacher,projected in loader:
        model.train()
        teacher=teacher.cuda(non_blocking=True).float()
        projected=projected.cuda(non_blocking=True).float() if a.variant=="projected" else None
        rate=.002*(step+1)/100 if step<100 else .0002+.5*.0018*(1+math.cos(math.pi*(step-100)/(a.steps-100)))
        for group in optimizer.param_groups:group["lr"]=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            decoded=active(teacher,projected)
            loss=loss_fn(teacher,decoded,step)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,foreach=True)
        optimizer.step()
        step+=1;losses.append(loss.detach())
        if step%100==0 or step==a.stop:
            values=torch.stack(losses).float()
            if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(grad)):
                raise FloatingPointError(f"nonfinite training at {step}")
            now=time.time()
            log_event(run/"training.jsonl",event="progress",step=step,loss=float(values.mean()),gradient=float(grad),
                lr=rate,seconds=now-timer,steps_sec=(step-last_step)/(now-last_time),
                gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses=[];last_time=now;last_step=step
        if step%1000==0 or step==a.stop or step==2500:
            ck={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":args,"step":step,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
            torch.save(ck,resume_dir/"last.pt.tmp");(resume_dir/"last.pt.tmp").replace(resume_dir/"last.pt")
            if step==a.stop or (step==2500 and not a.resume_root):
                torch.save({"model":model.state_dict(),"config":args,"step":step},run/f"model_{step}.pt")
    (run/f"COMPLETE_{step}").write_text(str(step))
    log_event(run/"training.jsonl",event="complete",step=step,seconds=time.time()-timer)
if __name__=="__main__":main()
