"""Normal residual precision estimation for a fixed reverse-distillation model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.covariance import oas
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import DirectionalRD
from train import CachedFeatures

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ['residual_full','tangent_diagonal','tangent_full','tangent_joint']


def residual_vectors(teacher,output,tangent):
    residuals = []
    for h,student in enumerate(output['reconstruction']):
        target = F.normalize(teacher[:,h,5:].float(),dim=-1)
        mean = F.normalize(student[:,5:].float(),dim=-1)
        if tangent:
            cosine = (target*mean).sum(-1,keepdim=True).clamp(-1,1)
            perpendicular = target-cosine*mean
            sine = perpendicular.norm(dim=-1,keepdim=True)
            # Spherical logarithm, with its continuous zero-angle limit.
            angle = torch.atan2(sine,cosine)
            scale = torch.where(sine>1e-7,angle/sine.clamp_min(1e-7),torch.ones_like(sine))
            residual = perpendicular*scale
        else:
            residual = target-mean
        residuals.append(residual)
    return residuals


def residual_score(teacher,output,statistics,variant):
    vectors = residual_vectors(teacher,output,variant.startswith('tangent'))
    if variant == 'tangent_joint':
        vectors = [torch.cat(vectors,dim=-1)]
    scores = []
    for vector,stat in zip(vectors,statistics):
        centered = vector-stat['mean'].to(vector.device)
        whitening = stat['whitening'].to(vector.device)
        scores.append((centered@whitening).square().mean(-1))
    score = torch.stack(scores).mean(0)
    side = int(score.shape[-1]**.5)
    return score.reshape(-1,side,side)


def fit(dataset,seed,variant):
    torch.set_num_threads(2)
    source = ROOT/'results'/f'{dataset}_dinomaly_s{seed}'
    run = ROOT/'results'/f'{dataset}_{variant}_s{seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'COMPLETE').exists():
        return
    base = torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    model = DirectionalRD('dinomaly').cuda().eval()
    model.load_state_dict(base['model'])
    data = CachedFeatures(dataset,280,'train')
    samples = [[],[]]
    with torch.inference_mode():
        for teacher,ids in DataLoader(data,batch_size=32,num_workers=4,pin_memory=True):
            teacher = teacher.cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output = model(teacher)
            vectors = residual_vectors(teacher,output,variant.startswith('tangent'))
            positions = []
            for index in ids.tolist():
                digest = hashlib.sha256(('residual-fit:'+data.rows[index]['source_sha256']).encode()).hexdigest()
                generator = np.random.default_rng(int(digest[:16],16))
                positions.append(generator.choice(vectors[0].shape[1],8,replace=False))
            positions = torch.as_tensor(np.asarray(positions),device='cuda')
            batch = torch.arange(len(teacher),device='cuda')[:,None]
            for h,vector in enumerate(vectors):
                samples[h].append(vector[batch,positions].reshape(-1,384).cpu().numpy())
    statistics = []
    report = []
    if variant == 'tangent_joint':
        samples = [[np.concatenate([np.concatenate(arrays) for arrays in samples],axis=-1)]]
    for arrays in samples:
        values = np.concatenate(arrays).astype(np.float64)
        mean = values.mean(0)
        covariance,shrinkage = oas(values,assume_centered=False)
        if variant=='tangent_diagonal':
            covariance = np.diag(np.diag(covariance))
        eigenvalues,eigenvectors = np.linalg.eigh(covariance)
        assert eigenvalues.min()>0
        whitening = eigenvectors/np.sqrt(eigenvalues)[None]
        statistics.append({'mean':torch.from_numpy(mean).float(),'whitening':torch.from_numpy(whitening).float()})
        report.append({'normal_tokens':len(values),'dimension':values.shape[1],'shrinkage':float(shrinkage),
                       'log_determinant':float(np.log(eigenvalues).sum()),
                       'minimum_eigenvalue':float(eigenvalues.min()),'maximum_eigenvalue':float(eigenvalues.max())})
    validation = CachedFeatures(dataset,280,'validation')
    validation_nll = []
    with torch.inference_mode():
        for teacher,_ in DataLoader(validation,batch_size=32,num_workers=2):
            teacher = teacher.cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output = model(teacher)
            normalized_quadratic = residual_score(teacher,output,statistics,variant)
            dimension = sum(group['dimension'] for group in report)
            normalized_logdet = sum(group['log_determinant'] for group in report)/dimension
            nll = .5*(normalized_quadratic+normalized_logdet+np.log(2*np.pi))
            validation_nll.extend(nll.mean(dim=(-1,-2)).cpu().tolist())
    config = {**base['config'],'variant':variant,'base_variant':'dinomaly','residual_tokens_per_image':8,
              'covariance_estimator':'OAS','score':'mean squared whitened residual across groups and channels'}
    checkpoint = {'model':base['model'],'statistics':statistics,'config':config,'step':base['step']}
    torch.save(checkpoint,run/'model.pt')
    (run/'config.json').write_text(json.dumps(config,indent=2))
    (run/'split_ids.json').write_bytes((source/'split_ids.json').read_bytes())
    (run/'residual_fit.json').write_text(json.dumps({'groups':report,'normal_validation_nll_per_dimension':float(np.mean(validation_nll)),
        'base_checkpoint_sha256':hashlib.sha256((source/'model.pt').read_bytes()).hexdigest()},indent=2))
    (run/'COMPLETE').write_text(str(base['step']))
    print(json.dumps({'run':run.name,'groups':report,'normal_validation_nll_per_dimension':float(np.mean(validation_nll))}))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['3cad','mvtec_ad2'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--variant',choices=VARIANTS,default='tangent_full')
    args = parser.parse_args()
    fit(args.dataset,args.seed,args.variant)
