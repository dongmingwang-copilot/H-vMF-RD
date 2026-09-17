import sys,time,json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cache_features import load_encoder,encode_groups
from common import ROOT,WORK,CACHE,split_rows,log_event
class Images(Dataset):
    def __init__(self,rows,size):
        self.rows=rows
        self.transform=transforms.Compose([transforms.Resize((size,size),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]["image"]) as im:return self.transform(im.convert("RGB"))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--size",type=int,default=448);ap.add_argument("--batch",type=int,default=64);ap.add_argument("--dataset",default="both");args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    enc=load_encoder("cuda")
    for ds in (["mvtec_ad2","3cad"] if args.dataset=="both" else [args.dataset]):
        folder=CACHE/f"{ds}_{args.size}";folder.mkdir(exist_ok=True)
        train=split_rows(ds,"train");val=split_rows(ds,"validation");rows=train+val
        shape=(len(rows),2,(args.size//14)**2+5,384)
        meta={"dataset":ds,"size":args.size,"shape":shape,"train_count":len(train),"validation_count":len(val),"row_ids":[r["id"] for r in rows],"precision":"bfloat16 encoder; float16 cache","manifest_sha256":hashlib.sha256((ROOT/"data"/ds/"manifest.json").read_bytes()).hexdigest()}
        (folder/"metadata.json").write_text(json.dumps(meta))
        if (folder/"COMPLETE").exists():continue
        start=int((folder/"progress").read_text()) if (folder/"progress").exists() else 0
        out=np.memmap(folder/"features.dat",dtype=np.float16,mode="r+" if start else "w+",shape=shape)
        loader=DataLoader(Images(rows[start:],args.size),batch_size=args.batch,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)
        cursor=start;beg=time.time()
        with torch.inference_mode():
            for x in loader:
                with torch.autocast("cuda",dtype=torch.bfloat16):features=encode_groups(enc,x.cuda(non_blocking=True))
                y=features.cpu().to(torch.float16).numpy();assert np.isfinite(y).all()
                out[cursor:cursor+len(y)]=y;cursor+=len(y)
                if cursor%256<args.batch or cursor==len(rows):
                    out.flush();(folder/"progress").write_text(str(cursor))
                    log_event(WORK/"cache.jsonl",dataset=ds,done=cursor,total=len(rows),images_sec=(cursor-start)/(time.time()-beg),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
        (folder/"COMPLETE").write_text(str(cursor))
        log_event(WORK/"cache.jsonl",dataset=ds,event="complete",seconds=time.time()-beg)
if __name__=="__main__":main()
