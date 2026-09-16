"""RD and RD++ adapters with a common normal-only multicategory protocol."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import numba
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
numba.set_num_threads(1)


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/'code/third_party'/relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


resnet = module('rd4ad_resnet', 'rd4ad/resnet.py')
de_resnet = module('rd4ad_decoder', 'rd4ad/de_resnet.py')
rdpp = module('rdpp_utils', 'rdpp/utils/utils_train.py')
simplex = module('rdpp_noise', 'rdpp/dataset/noise.py')


def selected_rows(dataset, split):
    rows = json.loads((ROOT/'data'/dataset/'manifest.json').read_text())
    has_validation = any(r['split']=='validation' for r in rows)
    test_hashes = {r['source_sha256'] for r in rows if r['split'] in {'test','test_public'}}
    chosen = []
    for row in rows:
        held = int(hashlib.sha256(('normal-validation:'+row['source_sha256']).encode()).hexdigest()[:8],16)%10==0
        eligible = row['source_sha256'] not in test_hashes
        if split == 'train':
            use = eligible and row['split']=='train' and (has_validation or not held)
        elif split == 'validation':
            use = eligible and (row['split']=='validation' if has_validation else row['split']=='train' and held)
        else:
            use = row['split'] in {'test','test_public'}
        if use:
            chosen.append(row)
    if split != 'test':
        assert all(r['label']==0 for r in chosen)
    return chosen


class BaselineImages(Dataset):
    def __init__(self, rows, noise=False, seed=17):
        self.rows, self.noise, self.seed = rows, noise, seed
        self.simplex = simplex.Simplex_CLASS() if noise else None
        self.cached_images = [self._read_image(row) for row in rows] if noise else None

    @staticmethod
    def _read_image(row):
        image = cv2.imread(row['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image.astype(np.float64) / 255., (256, 256), interpolation=cv2.INTER_LINEAR)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        step = 0
        if isinstance(index, tuple):
            index, step = index
        if self.cached_images is None:
            image = self._read_image(self.rows[index])
        else:
            image = self.cached_images[index]
        mean, std = np.array([.485,.456,.406]), np.array([.229,.224,.225])
        clean = torch.from_numpy(((image-mean)/std).transpose(2,0,1).copy()).float()
        if not self.noise:
            return clean,index
        rng = np.random.RandomState((self.seed*1000003+step*9176+index)%2**32)
        h,w = rng.randint(10,32,size=2)
        y,x = rng.randint(1,256-h),rng.randint(1,256-w)
        self.simplex.newSeed(int(rng.randint(1,2**31-1)))
        noise = self.simplex.rand_3d_octaves((3,h,w),6,.6).transpose(1,2,0)
        corrupted = image.copy()
        corrupted[y:y+h,x:x+w] += .2*noise
        noisy = torch.from_numpy(((corrupted-mean)/std).transpose(2,0,1).copy()).float()
        return clean,noisy,index


class BaselineBatches:
    def __init__(self,count,batch,seed,start,stop):
        from train import FixedStepBatches
        self.batches = FixedStepBatches(count,batch,seed,start,stop)
        self.start = start

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for step,batch in enumerate(self.batches,self.start):
            yield [(index,step) for index in batch]


class CNNReverse(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        self.variant = variant
        self.encoder,self.bn = resnet.wide_resnet50_2(pretrained=False)
        if pretrained:
            weights = ROOT/'pretrained/wide_resnet50_2-95faca4d.pth'
            weights.parent.mkdir(parents=True,exist_ok=True)
            if not weights.exists():
                torch.hub.download_url_to_file('https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',str(weights))
            state = torch.load(weights,map_location='cpu',weights_only=True)
            # The RD teacher exposes only the first three ResNet stages.
            filtered = {k:v for k,v in state.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered,strict=True)
        self.encoder.eval().requires_grad_(False)
        self.decoder = de_resnet.de_wide_resnet50_2(pretrained=False)
        self.projection = rdpp.MultiProjectionLayer(base=64) if variant=='rdpp' else nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            teacher = self.encoder(images)
        projected = self.projection(teacher)
        reconstruction = self.decoder(self.bn(projected))
        return teacher,reconstruction,projected

    def anomaly_map(self, images):
        teacher,student,_ = self(images)
        maps = [1-F.cosine_similarity(a.float(),b.float(),dim=1) for a,b in zip(teacher,student)]
        return torch.stack([F.interpolate(m[:,None],size=(64,64),mode='bilinear',align_corners=True)[:,0] for m in maps]).mean(0)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    run = ROOT/'results'/f'{args.dataset}_{args.variant}_s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'COMPLETE').exists():
        return
    rows = selected_rows(args.dataset,'train')
    data = BaselineImages(rows,noise=args.variant=='rdpp',seed=args.seed)
    model = CNNReverse(args.variant).cuda()
    groups = [{'params':list(model.bn.parameters())+list(model.decoder.parameters()),'lr':.005}]
    if args.variant=='rdpp':
        groups.append({'params':model.projection.parameters(),'lr':.001})
    optimizer = torch.optim.Adam(groups,betas=(.5,.999))
    projection_loss = rdpp.Revisit_RDLoss() if args.variant=='rdpp' else None
    step = 0
    if (run/'last.pt').exists():
        checkpoint = torch.load(run/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        torch.set_rng_state(checkpoint['rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint['cuda_rng']])
    # Simplex initializes GNU OpenMP/Numba state; using worker processes after
    # that initialization is unsafe on this runtime. The batch order and
    # seeded noise remain unchanged with main-process loading.
    workers = 0 if args.variant == 'rdpp' else 4
    loader = DataLoader(data,batch_sampler=BaselineBatches(len(data),args.batch,args.seed,step,args.steps),
                        num_workers=workers,pin_memory=True,
                        persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(args.seed))
    (run/'config.json').write_text(json.dumps(vars(args),indent=2))
    (run/'split_ids.json').write_text(json.dumps({'train':[r['id'] for r in rows],
         'validation':[r['id'] for r in selected_rows(args.dataset,'validation')]}))
    started = time.time()
    losses = []
    for batch in loader:
        model.train()
        clean = batch[0].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            teacher,student,projected = model(clean)
            loss = rdpp.loss_fucntion([a.float() for a in teacher],[b.float() for b in student])
            if args.variant=='rdpp':
                with torch.no_grad():
                    corrupted_teacher = model.encoder(batch[1].cuda(non_blocking=True))
                projected_corrupted = model.projection(corrupted_teacher)
        if args.variant=='rdpp':
            loss = loss + .2*projection_loss([a.float() for a in corrupted_teacher],
                   [a.float() for a in projected_corrupted],[a.float() for a in projected])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite baseline loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        optimizer.step()
        step += 1
        losses.append(float(loss.detach()))
        if step%100==0:
            record={'step':step,'loss':float(np.mean(losses)),'elapsed_seconds':time.time()-started}
            with (run/'training.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            losses=[]
        if step%1000==0 or step==args.steps:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},run/'last.pt.tmp')
            (run/'last.pt.tmp').replace(run/'last.pt')
    state = {k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}
    torch.save({'model':state,'config':vars(args),'step':step},run/'model.pt')
    (run/'COMPLETE').write_text(str(step))
    (run/'last.pt').unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['3cad','mvtec_ad2'])
    parser.add_argument('--variant',required=True,choices=['rd','rdpp'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--batch',type=int,default=16)
    train(parser.parse_args())
