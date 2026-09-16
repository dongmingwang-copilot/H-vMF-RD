"""Generate manuscript tables exclusively from complete measured results."""
import json
from pathlib import Path
import numpy as np
from study import METHODS,METRICS,DATASETS,PRIMARY,ABLATIONS,seeds_for

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper/results'
OUT.mkdir(parents=True,exist_ok=True)


def read(source,variant,target,seeds):
    paths=[ROOT/'results'/f'{source}_{variant}_s{seed}'/f'eval_{target}/metrics.json' for seed in seeds]
    if any(not path.exists() for path in paths):
        return None
    return np.array([[json.loads(path.read_text())['mean'][metric]*100 for metric in METRICS] for path in paths])


def cells(values,best=None):
    mean=values.mean(0)
    std=values.std(0,ddof=1) if len(values)>1 else None
    result=[]
    for index,value in enumerate(mean):
        formatted=f'{value:.2f}' if std is None else f'{value:.2f}' + r'\,$\pm$\,' + f'{std[index]:.2f}'
        if best is not None and abs(value-best[index])<1e-8:
            formatted=r'\textbf{'+formatted+'}'
        result.append(formatted)
    return ' & '.join(result)


def full_table(data,caption,label,headers):
    best=[np.max(np.array([pair[i].mean(0) for _,pair in data]),axis=0) for i in range(2)]
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
           r'\setlength{\tabcolsep}{3.5pt}',r'\footnotesize',r'\begin{tabular}{lcccccccc}',r'\toprule',
           r' & \multicolumn{4}{c}{'+headers[0]+r'} & \multicolumn{4}{c}{'+headers[1]+r'} \\',
           r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
           r'Method & I-AUROC & P-AUROC & P-AP & AUPRO & I-AUROC & P-AUROC & P-AP & AUPRO \\',r'\midrule']
    for name,pair in data:
        lines.append(name+' & '+cells(pair[0],best[0])+' & '+cells(pair[1],best[1])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    return '\n'.join(lines)+'\n'


def comparison():
    data=[]
    for variant,name in METHODS:
        seeds=seeds_for(variant)
        pair=[read(dataset,variant,dataset,seeds) for dataset in DATASETS]
        if any(values is None for values in pair):
            (OUT/'comparisons.tex').unlink(missing_ok=True)
            return False
        data.append((name,pair))
    text=full_table(data,'Within-dataset comparison on complete public test partitions. Values are percentages. '
        'Dinomaly and '+r'\method{}'+' report mean and sample standard deviation over three seeds; other comparisons use seed 17. '
        'AUPRO integrates to FPR 0.30 on 3CAD and 0.05 on MVTec AD 2.','tab:comparison',['3CAD','MVTec AD 2 (public)'])
    for i,dataset in enumerate(['3CAD','MVTec AD 2']):
        baseline=data[-2][1][i].mean(0)
        ours=data[-1][1][i].mean(0)
        delta=ours-baseline
        text += (f'On {dataset}, '+r'\method{}'+f' obtains {ours[0]:.2f} I-AUROC, {ours[1]:.2f} P-AUROC, '
            f'{ours[2]:.2f} P-AP, and {ours[3]:.2f} AUPRO. Relative to the matched Dinomaly baseline, the changes are '
            +', '.join(f'{d:+.2f}' for d in delta)+' percentage points, respectively.\n\n')
    (OUT/'comparisons.tex').write_text(text)
    return True


def ablation():
    data=[]
    for variant,name in ABLATIONS:
        pair=[read(d,variant,d,[17]) for d in DATASETS]
        if any(v is None for v in pair):
            (OUT/'ablations.tex').unlink(missing_ok=True)
            return False
        data.append((name,pair))
    text=full_table(data,'Reconstruction ablations with a common teacher and decoder and seed 17. Values are percentages.',
                    'tab:ablation',['3CAD','MVTec AD 2 (public)'])
    single = data[1][1]
    cosine = data[0][1]
    delta = [single[i][0]-cosine[i][0] for i in range(2)]
    text += (f'The single-component likelihood changes P-AP and AUPRO by {delta[0][2]:+.2f} and '
             f'{delta[0][3]:+.2f} percentage points on 3CAD, and by {delta[1][2]:+.2f} and '
             f'{delta[1][3]:+.2f} points on MVTec AD 2.\n\n')
    for index,dataset in enumerate(['3CAD','MVTec AD 2']):
        cosine_change = data[2][1][index][0]-cosine[index][0]
        vmf_change = data[3][1][index][0]-single[index][0]
        text += (f'On {dataset}, adding directional denoising to cosine reconstruction changes P-AP and AUPRO by '
                 f'{cosine_change[2]:+.2f} and {cosine_change[3]:+.2f} points. '
                 'With the single-component likelihood, the corresponding changes are '
                 f'{vmf_change[2]:+.2f} and {vmf_change[3]:+.2f} points.\n\n')
    text += (
             'The additional mixture components do not yield a consistent improvement. '
             'Sharing the latent component across hierarchies also does not improve both datasets.\n')
    (OUT/'ablations.tex').write_text(text)
    return True


def transfer():
    data=[]
    for variant,name in METHODS:
        seeds=seeds_for(variant)
        pair=[read('3cad',variant,'mvtec_ad2',seeds),read('mvtec_ad2',variant,'3cad',seeds)]
        if any(v is None for v in pair):
            (OUT/'transfer.tex').unlink(missing_ok=True)
            return False
        data.append((name,pair))
    text=full_table(data,'Direct bidirectional transfer without target-domain training. Values are percentages. '
        'Dinomaly and '+r'\method{}'+' report three-seed means and sample standard deviations; other methods use seed 17. '
        'The AUPRO integration limit follows the target dataset.','tab:transfer',
        [r'3CAD $\rightarrow$ MVTec AD 2',r'MVTec AD 2 $\rightarrow$ 3CAD'])
    (OUT/'transfer.tex').write_text(text)
    return True


if __name__=='__main__':
    print(json.dumps({'comparisons':comparison(),'ablations':ablation(),'transfer':transfer()}))
