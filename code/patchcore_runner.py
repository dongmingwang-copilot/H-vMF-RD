"""PatchCore using official feature extraction and approximate greedy coreset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2

from baselines import BaselineImages, selected_rows

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code/third_party/patchcore'))
from patchcore.patchcore import PatchCore
from patchcore.sampler import ApproximateGreedyCoresetSampler
from patchcore.common import FaissNN


def make_model():
    backbone = wide_resnet50_2(weights=None)
    weights=ROOT/'pretrained/wide_resnet50_2-95faca4d.pth'
    weights.parent.mkdir(parents=True,exist_ok=True)
    if not weights.exists():
        torch.hub.download_url_to_file('https://download.pytorch.org/models/'+weights.name,str(weights))
    backbone.load_state_dict(torch.load(weights,weights_only=True))
    backbone.name = 'wide_resnet50_2'
    backbone.eval().requires_grad_(False)
    model = PatchCore(torch.device('cuda'))
    model.load(backbone,['layer2','layer3'],torch.device('cuda'),(3,256,256),1024,1024,
               patchsize=3,anomaly_score_num_nn=1,nn_method=FaissNN(False,8))
    model.eval()
    return model


def fit(dataset,seed,candidate_limit,memory_size):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    run=ROOT/'results'/f'{dataset}_patchcore_s{seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'COMPLETE').exists():
        return
    model=make_model()
    rows=selected_rows(dataset,'train')
    total_patches=len(rows)*32*32
    chosen=np.sort(np.random.choice(total_patches,min(candidate_limit,total_patches),replace=False))
    loader=DataLoader(BaselineImages(rows),batch_size=16,num_workers=4,pin_memory=True)
    candidates=[]
    cursor=0
    with torch.inference_mode():
        for images,_ in loader:
            features=model._embed(images.cuda(),detach=False)
            count=len(features)
            indices=chosen[(chosen>=cursor)&(chosen<cursor+count)]-cursor
            candidates.append(features[torch.as_tensor(indices,device='cuda')].cpu().numpy())
            cursor+=count
            if cursor%(1024*256)==0:
                print('PatchCore candidate extraction',cursor//1024,'/',len(rows),flush=True)
    assert cursor==total_patches
    candidates=np.concatenate(candidates)
    assert len(candidates)==len(chosen)
    sampler=ApproximateGreedyCoresetSampler(min(memory_size/len(candidates),.999999),torch.device('cuda'))
    with torch.no_grad():
        memory=sampler.run(candidates)
    np.save(run/'memory.npy',memory)
    (run/'config.json').write_text(json.dumps({'variant':'patchcore','dataset':dataset,'seed':seed,
         'input_size':256,'candidate_limit':candidate_limit,'memory_size':len(memory),
         'backbone':'wide_resnet50_2','layers':['layer2','layer3'],'patchsize':3,
         'pretrain_embed_dimension':1024,'target_embed_dimension':1024,
         'upstream_revision':'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'},indent=2))
    (run/'split_ids.json').write_text(json.dumps({'train':[r['id'] for r in rows]}))
    (run/'COMPLETE').write_text(str(len(memory)))
    print('PatchCore memory complete',memory.shape,flush=True)


def predict(run,target,output):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model=make_model()
    memory=np.load(run/'memory.npy')
    memory_tensor = torch.from_numpy(memory).cuda()
    rows=selected_rows(target,'test')
    loader=DataLoader(BaselineImages(rows),batch_size=16,num_workers=4,pin_memory=True)
    maps,scores=[],[]
    started=time.time()
    with torch.inference_mode():
        for images,_ in loader:
            features=model._embed(images.cuda(),detach=False)
            distances=nearest_squared_l2(features,memory_tensor).cpu().numpy().reshape(len(images),32,32)
            maps.extend(distances)
            # Official PatchCore image score is the maximum patch distance.
            scores.extend(distances.reshape(len(images),-1).max(1))
            if len(maps)%256==0:
                print('PatchCore prediction',len(maps),'/',len(rows),'seconds',time.time()-started,flush=True)
    np.savez_compressed(output/'predictions.npz',maps=np.asarray(maps,dtype=np.float32),scores=np.asarray(scores))
    (output/'rows.json').write_text(json.dumps(rows))
    (output/'prediction_metadata.json').write_text(json.dumps({'size':256,'gaussian_sigma':4.,
        'image_score':'maximum patch distance','stored_maps':'raw patch maps','target':target,
        'nearest_neighbor':'exhaustive float32 squared L2 on GPU, TF32 disabled',
        'memory_sha256':hashlib.sha256((run/'memory.npy').read_bytes()).hexdigest()},indent=2))


def nearest_squared_l2(queries,memory,chunk=2048):
    memory_norm = memory.square().sum(-1)[None]
    values = []
    for query in queries.split(chunk):
        distances = query.square().sum(-1)[:,None]+memory_norm-2*(query@memory.T)
        values.append(distances.clamp_min_(0).amin(dim=1))
    return torch.cat(values)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['3cad','mvtec_ad2'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--candidates',type=int,default=100000)
    parser.add_argument('--memory',type=int,default=10000)
    args=parser.parse_args()
    fit(args.dataset,args.seed,args.candidates,args.memory)
