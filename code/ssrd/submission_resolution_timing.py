from pathlib import Path
import subprocess,json,sys,numpy as np
from common import ROOT,WORK
out=WORK/"submission_case_study"/"resolution_timing";out.mkdir(exist_ok=True)
records=[]
for repeat in range(3):
 for batch in ([1,16] if repeat%2==0 else [16,1]):
  path=out/f"ssrd896_b{batch}_r{repeat}.json"
  subprocess.run([sys.executable,str(ROOT/"code/ssrd/benchmark_resolution.py"),"--batch",str(batch),"--output",str(path)],check=True)
  print(f"COMPLETE round={repeat} batch={batch}",flush=True)
for batch in [1,16]:
 rr=[json.loads((out/f"ssrd896_b{batch}_r{r}.json").read_text()) for r in range(3)]
 sec=np.array([t for r in rr for t in r["seconds"]])
 records.append({"method":"spatial_highres","size":896,"batch":batch,"median_batch_ms":float(np.median(sec)*1000),"p95_batch_ms":float(np.quantile(sec,.95)*1000),"images_sec":float(batch/sec.mean()),"peak_allocated_mb":max(r["peak_allocated_mb"] for r in rr),"trainable_parameters":rr[0]["trainable_parameters"],"checkpoint_sha256":rr[0]["checkpoint_sha256"],"round_medians_ms":[float(np.median(r["seconds"])*1000) for r in rr]})
report={"size":896,"sigma":8,"rounds":3,"warmup_batches":50,"measured_batches_per_round":100,"timed":"teacher, decoder, score map, D2H, resize, Gaussian and image maximum; pre-normalized GPU-resident inputs","records":records}
(out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
