from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"results"
CACHE=Path(os.environ.get("SSRD_CACHE_ROOT",str(ROOT/"cache")))
WORK.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
def split_rows(dataset,split):
    rows=json.loads((ROOT/"data"/dataset/"manifest.json").read_text())
    has_val=any(r["split"]=="validation" for r in rows)
    test_hashes={r["source_sha256"] for r in rows if r["split"] in {"test","test_public"}}
    result=[]
    for row in rows:
        held=int(hashlib.sha256(("normal-validation:"+row["source_sha256"]).encode()).hexdigest()[:8],16)%10==0
        eligible=row["source_sha256"] not in test_hashes
        use=(eligible and row["split"]=="train" and (has_val or not held)) if split=="train" else (eligible and (row["split"]=="validation" if has_val else row["split"]=="train" and held)) if split=="validation" else row["split"] in {"test","test_public"}
        if use:result.append(row)
    if split!="test":assert all(r["label"]==0 for r in result)
    return result
def log_event(path,**values):
    values={"time":time.time(),**values}
    with Path(path).open("a") as f:f.write(json.dumps(values)+"\n")
    print(json.dumps(values),flush=True)
def cache_data(dataset,size=448):
    path=CACHE/f"{dataset}_{size}"
    meta=json.loads((path/"metadata.json").read_text())
    assert (path/"COMPLETE").exists()
    data=np.memmap(path/"features.dat",dtype=np.float16,mode="r",shape=tuple(meta["shape"]))
    return path,meta,data
