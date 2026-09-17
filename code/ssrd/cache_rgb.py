"""Cache exactly the training image transform; no new augmentation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,os,time,hashlib
import numpy as np
from PIL import Image
from common import CACHE,WORK,split_rows
def decode(path):
    with Image.open(path) as im:return np.asarray(im.convert("RGB").resize((448,448),Image.Resampling.BICUBIC),dtype=np.uint8)
def main():
    for ds in ["mvtec_ad2","3cad"]:
        folder=CACHE/f"{ds}_rgb";folder.mkdir(exist_ok=True)
        rows=split_rows(ds,"train")
        if (folder/"COMPLETE").exists():continue
        shape=(len(rows),448,448,3)
        f=np.memmap(folder/"rgb.dat",dtype=np.uint8,mode="w+",shape=shape)
        start=time.time()
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,image in enumerate(pool.map(decode,[r["image"] for r in rows],chunksize=8)):
                f[i]=image
        f.flush()
        (folder/"metadata.json").write_text(json.dumps({"shape":shape,"row_ids":[r["id"] for r in rows],"transform":"PIL RGB direct 448x448 BICUBIC uint8; identical to TrainingPairs"}))
        (folder/"COMPLETE").write_text(str(len(rows)))
        print(json.dumps({"dataset":ds,"count":len(rows),"seconds":time.time()-start}),flush=True)
if __name__=="__main__":main()
