"""Scientific figures generated from measured predictions and metric JSON."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
from study import PRIMARY,METHOD_NAME

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})


def save(fig,name):
    for extension in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+extension),bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def category_scatter():
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.1),layout='constrained')
    for dataset,color,marker,label in [('3cad','#4477AA','o','3CAD'),('mvtec_ad2','#CC6677','^','MVTec AD 2')]:
        values={}
        for variant in ['dinomaly',PRIMARY]:
            paths=[ROOT/'results'/f'{dataset}_{variant}_s{seed}'/f'eval_{dataset}/metrics.json' for seed in [17,29,43]]
            if not all(path.exists() for path in paths):
                plt.close(fig)
                return False
            reports=[json.loads(path.read_text())['categories'] for path in paths]
            values[variant]={category:np.array([[next(r for r in report if r['category']==category)[m]*100
                        for m in ['p_ap','aupro']] for report in reports]) for category in [r['category'] for r in reports[0]]}
        for i,ax in enumerate(axes):
            for j,category in enumerate(values['dinomaly']):
                x=values['dinomaly'][category][:,i]
                y=values[PRIMARY][category][:,i]
                ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),
                    fmt=marker,color=color,markersize=4.5,elinewidth=.6,capsize=1.5,
                    alpha=.85,label=label if j==0 else None)
    for ax,title in zip(axes,['Pixel average precision','Per-region overlap AUC']):
        ax.plot([0,100],[0,100],ls='--',lw=.8,color='#999999',zorder=0)
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='Dinomaly (%)',ylabel=METHOD_NAME+' (%)',title=title)
        ax.set_aspect('equal')
        ax.legend(frameon=False,loc='lower right')
        ax.grid(alpha=.15,lw=.5)
    save(fig,'category_comparison')
    return True


def qualitative():
    specs=[('3cad','Aluminum_Camera_Cover'),('3cad','Copper_Stator'),('mvtec_ad2','fabric'),('mvtec_ad2','vial')]
    predictions={}
    for dataset in ['3cad','mvtec_ad2']:
        for variant in ['dinomaly',PRIMARY]:
            folder=ROOT/'results'/f'{dataset}_{variant}_s17'/f'eval_{dataset}'
            if not (folder/'prediction_metadata.json').exists():
                return False
            data=np.load(folder/'predictions.npz')
            rows=json.loads((folder/'rows.json').read_text())
            maps=data['maps']
            low,high=np.percentile(maps,[1,99.5])
            predictions[(dataset,variant)]={'maps':maps,'rows':{r['id']:(i,r) for i,r in enumerate(rows)},
                                            'normalization':Normalize(low,high,clip=True)}
    fig,axes=plt.subplots(4,4,figsize=(7.16,6.5),gridspec_kw={'wspace':.035,'hspace':.17})
    selected=[]
    for index,(dataset,category) in enumerate(specs):
        reference=predictions[(dataset,'dinomaly')]
        candidates=sorted([r for _,r in reference['rows'].values() if r['category']==category and r['label']],key=lambda r:r['id'])
        row=candidates[0]
        for candidate in candidates:
            mask=np.asarray(Image.open(candidate['mask']))>0
            if .005<=mask.mean()<=.20:
                row=candidate
                break
        selected.append({'dataset':dataset,'category':category,'id':row['id']})
        image=np.asarray(Image.open(row['image']).convert('RGB').resize((280,280)))
        mask=np.asarray(Image.open(row['mask']).resize((280,280),Image.Resampling.NEAREST))>0
        axes[index,0].imshow(image)
        axes[index,1].imshow(mask,cmap='gray',vmin=0,vmax=1)
        for column,variant in enumerate(['dinomaly',PRIMARY],2):
            data=predictions[(dataset,variant)]
            position,_=data['rows'][row['id']]
            anomaly=cv2.resize(data['maps'][position],(280,280),interpolation=cv2.INTER_LINEAR)
            anomaly=cv2.GaussianBlur(anomaly,(0,0),sigmaX=4.,borderType=cv2.BORDER_REFLECT)
            axes[index,column].imshow(anomaly,cmap='magma',norm=data['normalization'])
        axes[index,0].text(-.06,.5,category.replace('Aluminum_','Al. ').replace('_',' '),
                           transform=axes[index,0].transAxes,ha='right',va='center',rotation=90,fontsize=7)
        for ax in axes[index]:
            ax.axis('off')
    for ax,title in zip(axes[0],['Image','Ground truth','Dinomaly',METHOD_NAME]):
        ax.set_title(title,pad=5)
    (OUT/'qualitative_selection.json').write_text(json.dumps({'selection':selected,
        'rule':'First sorted anomalous identifier with mask fraction 0.005-0.20, falling back to first anomalous identifier.',
        'color_limits':'Fixed 1st and 99.5th percentiles of all public test token scores, per method and dataset.'},indent=2))
    save(fig,'qualitative')
    return True


if __name__=='__main__':
    print(json.dumps({'category_comparison':category_scatter(),'qualitative':qualitative()}))
