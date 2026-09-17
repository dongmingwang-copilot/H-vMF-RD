"""Exact bin assignment acceleration for the existing ADEval histogram."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import VectorizedStatCurve
def indices_exact(preds,bins):
    # ADEval has -inf/+inf guard bins and float32 uniformly generated interior edges.
    if not np.all(np.diff(bins[1:-1])>0):
        return np.searchsorted(bins,preds,side="right")-1
    n=len(bins)-3
    index=np.floor((preds.astype(np.float64)-float(bins[1]))*(n/(float(bins[-2])-float(bins[1])))).astype(np.int64)+1
    np.clip(index,0,len(bins)-2,out=index)
    # Float32 edge rounding can move the arithmetic estimate across a boundary.
    for _ in range(4):
        low=preds<bins[index];high=preds>=bins[index+1]
        if not (low.any() or high.any()):return index
        index-=low;index+=high
    return np.searchsorted(bins,preds,side="right")-1
def accum(self,preds,label,weight=1.):
    preds=np.asarray(preds).reshape(-1);label=np.asarray(label).reshape(-1)!=0
    if not np.isfinite(preds).all():raise ValueError("Nonfinite prediction")
    indices=indices_exact(preds,self.bins)
    positive,negative=indices[label],indices[~label]
    pos=np.bincount(positive,minlength=len(self.pos_stat));neg=np.bincount(negative,minlength=len(self.neg_stat))
    self.pos_stat+=pos;self.neg_stat+=neg
    if np.ndim(weight)==0:
        self.weighted_pos_stat+=pos*float(weight);self.weighted_neg_stat+=neg*float(weight)
    else:
        weights=np.asarray(weight,dtype=np.float64).reshape(-1)
        self.weighted_pos_stat+=np.bincount(positive,weights=weights[label],minlength=len(pos))
        self.weighted_neg_stat+=np.bincount(negative,weights=weights[~label],minlength=len(neg))
def apply():VectorizedStatCurve.accum=accum
if __name__=="__main__":
    import time,json
    from common import WORK,log_event
    rng=np.random.default_rng(17)
    checks=[]
    for lo,hi in [(-.1,1.2),(.0137,.3801),(.45,.451),(.4,.40001)]:
        curve=VectorizedStatCurve(lo,hi,nstrips=262144)
        x=np.concatenate([rng.uniform(lo-.01,hi+.01,500000).astype(np.float32),
            curve.bins[1:-1],np.nextafter(curve.bins[1:-1],np.float32(-np.inf)),
            np.nextafter(curve.bins[1:-1],np.float32(np.inf))])
        start=time.time();old=np.searchsorted(curve.bins,x,side="right")-1;old_time=time.time()-start
        start=time.time();new=indices_exact(x,curve.bins);new_time=time.time()-start
        assert np.array_equal(old,new)
        checks.append({"range":[lo,hi],"values":len(x),"searchsorted_sec":old_time,"new_sec":new_time})
    # Compare the full weighted histogram, including region weights.
    a=VectorizedStatCurve(-.1,1.2,nstrips=262144);b=VectorizedStatCurve(-.1,1.2,nstrips=262144)
    x=rng.uniform(-.1,1.2,1000000).astype(np.float32);y=rng.random(len(x))<.04;weights=rng.uniform(.001,.2,len(x))
    a.accum(x,y,weights);accum(b,x,y,weights)
    assert all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ["pos_stat","neg_stat","weighted_pos_stat","weighted_neg_stat"])
    log_event(WORK/"profile.jsonl",event="metric_equivalence",checks=checks,weighted_histograms_identical=True)
