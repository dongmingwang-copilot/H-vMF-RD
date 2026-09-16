"""Measured normal-validation outputs for the denoising ablation figure."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from model import DirectionalRD,directional_patch_corruption
from train import CachedFeatures
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research/model_outputs'
OUT.mkdir(parents=True,exist_ok=True)

def error(target,output):
    return torch.stack([1-F.cosine_similarity(target[:,h,5:].float(),head[0][:,:,0].float(),dim=-1)
                        for h,head in enumerate(output['heads'])]).mean(0)

def main():
    torch.set_num_threads(2)
    expected=json.loads((ROOT/'research/denoising_p25_validation.json').read_text())['normal_validation']
    summary={}
    for dataset in ['3cad','mvtec_ad2']:
        data=CachedFeatures(dataset,280,'validation')
        saved={};summary[dataset]={}
        for variant in ['vmf_single','denoising_vmf_p25']:
            path=ROOT/'results'/f'{dataset}_{variant}_s17/model.pt'
            model=DirectionalRD(variant).cuda().eval()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
            generator=torch.Generator(device='cuda').manual_seed(9017)
            clean_values=[];corrupt_values=[];concentration=[]
            first=True
            with torch.inference_mode():
                for teacher,ids in DataLoader(data,batch_size=16,num_workers=2):
                    teacher=teacher.cuda().float()
                    corrupted=directional_patch_corruption(teacher,generator)
                    changed=(corrupted[:,0,5:]-teacher[:,0,5:]).abs().sum(-1)>1e-5
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        clean=model(teacher);altered=model(corrupted)
                    ec=error(teacher,clean);en=error(teacher,altered)
                    clean_values.append(ec.cpu().numpy())
                    corrupt_values.append(en[changed].cpu().numpy())
                    concentration.append(torch.stack([q[1][:,0] for q in clean['heads']],1).cpu().numpy()/384.)
                    if first:
                        saved['sample_mask']=changed[0].reshape(20,20).cpu().numpy()
                        saved[variant+'_sample_clean']=ec[0].reshape(20,20).cpu().numpy()
                        saved[variant+'_sample_corrupt']=en[0].reshape(20,20).cpu().numpy()
                        saved['sample_id']=np.array(data.rows[int(ids[0])]['id'])
                        saved['sample_image']=np.array(data.rows[int(ids[0])]['image'])
                        first=False
            ev=np.concatenate(clean_values).astype(np.float64)
            nv=np.concatenate(corrupt_values).astype(np.float64)
            kappas=np.concatenate(concentration)
            assert abs(ev.mean()-expected[dataset][variant]['clean_error'])<2e-6
            assert abs(nv.mean()-expected[dataset][variant]['corrupted_region_error'])<2e-6
            saved[variant+'_clean_per_image']=ev.mean(1)
            saved[variant+'_kappa']=kappas
            summary[dataset][variant]={'clean_error':float(ev.mean()),'corrupted_region_error':float(nv.mean()),
                'concentration_near_upper_bound_fraction':(kappas>15.9).mean(0).tolist(),
                'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'validation_images':len(data)}
            print(dataset,variant,summary[dataset][variant],flush=True)
            del model
        np.savez_compressed(OUT/f'{dataset}_validation.npz',**saved)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
