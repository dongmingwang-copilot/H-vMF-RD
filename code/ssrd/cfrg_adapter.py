"""Pinned CFRG architecture with common-protocol I/O and equivalent batched scoring."""
from pathlib import Path
import importlib.util,sys,types,json,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT

UPSTREAM=ROOT/"code/third_party/cfrg/upstream"
def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

# Isolate the architecture from the upstream training package's optional evaluator.
if "cfrg_vendor" not in sys.modules:
    package=types.ModuleType("cfrg_vendor")
    package.__path__=[str(UPSTREAM/"train_utils/toolbox/models/cfrg")]
    sys.modules["cfrg_vendor"]=package
from cfrg_vendor import resnet
_original_loader=resnet.load_state_dict_from_url
def cached_teacher(url,*args,**kwargs):
    local=ROOT/"pretrained"/url.rsplit("/",1)[-1]
    if local.exists():return torch.load(local,map_location="cpu",weights_only=True)
    return _original_loader(url,*args,**kwargs)
resnet.load_state_dict_from_url=cached_teacher
from cfrg_vendor.model import cfrg_net

def build_model():
    return cfrg_net()

def trainable_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith("teacher.")}

def load_trainable(model,state):
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith("teacher.") for k in result.missing_keys)

def objective(outputs,mask):
    normal,altered,student,reconstruction,logits=outputs
    recovery=sum((1-F.cosine_similarity(a.float().flatten(1),b.float().flatten(1),dim=1)).mean()
                 for a,b in zip(normal,reconstruction))
    distinction=0.
    for a,b in zip(altered,student):
        residual=1-F.cosine_similarity(a.float(),b.float(),dim=1)
        residual=F.interpolate(residual[:,None],size=mask.shape[-2:],mode="bilinear",align_corners=False)
        distinction=distinction+((1-mask)*residual+mask*(1-residual)).mean()
    distinction=distinction/3
    segmentation=F.binary_cross_entropy_with_logits(logits.float(),mask.float())
    return recovery+distinction+segmentation,torch.stack([recovery,distinction,segmentation])

def score_map(teacher,reconstruction,logits,size=448):
    residuals=[1-F.cosine_similarity(a.float(),b.float(),dim=1) for a,b in zip(teacher,reconstruction)]
    result=sum(F.interpolate(a[:,None],size=(size,size),mode="bilinear",align_corners=True)[:,0] for a in residuals)
    return result+F.interpolate(logits.float().sigmoid(),size=(size,size),mode="bilinear",align_corners=True)[:,0]

def upstream_augmentation():
    # imgaug 0.4 uses NumPy's removed dtype registry; the actual RNG and transform remain upstream.
    if not hasattr(np,"sctypes"):
        np.sctypes={"float":[np.float16,np.float32,np.float64],"int":[np.int8,np.int16,np.int32,np.int64],
                    "uint":[np.uint8,np.uint16,np.uint32,np.uint64],"complex":[np.complex64,np.complex128],
                    "others":[bool,object,str,bytes]}
    return load_file("cfrg_augmentation",UPSTREAM/"train_utils/toolbox/datasets/data_utils.py")

def verify_source():
    source=json.loads((UPSTREAM.parent/"SOURCE.json").read_text())
    for item in source["files"]:
        assert hashlib.sha256((UPSTREAM/item["path"]).read_bytes()).hexdigest()==item["sha256"],item["path"]
    return source["commit"]
