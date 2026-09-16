"""Normal-only gate for structured directional denoising development."""
import json
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import DirectionalRD,directional_patch_corruption
from train import CachedFeatures

ROOT = Path(__file__).resolve().parent.parent


def angular_error(teacher,output):
    if 'heads' in output:
        student = [head[0][:,:,0] for head in output['heads']]
    else:
        student = [value[:,5:] for value in output['reconstruction']]
    return torch.stack([1-F.cosine_similarity(teacher[:,h,5:].float(),value.float(),dim=-1)
                        for h,value in enumerate(student)]).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p25',action='store_true')
    args = parser.parse_args()
    pairs = [('denoising_cosine','dinomaly'),('denoising_vmf','vmf_single')]
    if args.p25:
        original = json.loads((ROOT/'research/denoising_validation.json').read_text())['gate']
        pairs = [(variant+'_p25',base) for variant,base in pairs if not original[variant]]
    variants = list(dict.fromkeys([base for _,base in pairs]+[variant for variant,_ in pairs]))
    torch.set_num_threads(2)
    report = {}
    for dataset in ['3cad','mvtec_ad2']:
        data = CachedFeatures(dataset,280,'validation')
        report[dataset] = {}
        for variant in variants:
            path = ROOT/'results'/f'{dataset}_{variant}_s17/model.pt'
            if not path.exists():
                print('Normal development validation pending',dataset,variant)
                return
            model = DirectionalRD(variant).cuda().eval()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
            generator = torch.Generator(device='cuda').manual_seed(9017)
            sums,counts = [0.,0.],[0,0]
            with torch.inference_mode():
                for teacher,_ in DataLoader(data,batch_size=16,num_workers=2):
                    teacher = teacher.cuda().float()
                    corrupted = directional_patch_corruption(teacher,generator)
                    changed = (corrupted[:,0,5:]-teacher[:,0,5:]).abs().sum(-1)>1e-5
                    for index,features in enumerate([teacher,corrupted]):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            output = model(features)
                        error = angular_error(teacher,output)
                        selected = error if index==0 else error[changed]
                        sums[index] += float(selected.double().sum())
                        counts[index] += selected.numel()
            report[dataset][variant] = {'clean_error':sums[0]/counts[0],'corrupted_region_error':sums[1]/counts[1]}
            del model
    gate = {}
    for variant,base in pairs:
        gate[variant] = all(report[d][variant]['clean_error']<=1.1*report[d][base]['clean_error']
                            and report[d][variant]['corrupted_region_error']<report[d][base]['corrupted_region_error']
                            for d in report)
    result = {'normal_validation':report,'gate':gate,'test_evaluation_authorized_by_protocol':[v for v,g in gate.items() if g]}
    filename = 'denoising_p25_validation.json' if args.p25 else 'denoising_validation.json'
    (ROOT/'research'/filename).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
