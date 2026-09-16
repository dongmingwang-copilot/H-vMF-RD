"""Dense scientific figures from fixed checkpoints and stored predictions."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from study import PRIMARY,DATASETS,SEEDS
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7,
 'axes.titlesize':7.4,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.4,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320})
BLUE='#4477AA';PURPLE='#AA3377';GREEN='#228833';GRAY='#6C737A'
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.04,facecolor='white')
    plt.close(fig)

def denoising_evidence():
    reports=[json.loads((ROOT/'research'/name).read_text())['normal_validation']
             for name in ['denoising_validation.json','denoising_p25_validation.json']]
    fig=plt.figure(figsize=(7.16,2.72))
    gs=fig.add_gridspec(2,5,width_ratios=[1.4,1.4,1.4,.65,2.6],wspace=.13,hspace=.60,
                       left=.035,right=.97,bottom=.14,top=.89)
    summary={}
    for row,dataset in enumerate(DATASETS):
        a=np.load(ROOT/'research/model_outputs'/f'{dataset}_validation.npz')
        e0=a['vmf_single_sample_corrupt'];e1=a[PRIMARY+'_sample_corrupt']
        upper=max(float(e0.max()),float(e1.max()))
        maps=[]
        for j in range(3):
            ax=fig.add_subplot(gs[row,j])
            if j==0:
                ax.imshow(Image.open(str(a['sample_image'])).convert('RGB').resize((280,280),Image.Resampling.BICUBIC))
                m=cv2.resize(a['sample_mask'].astype(np.uint8),(280,280),interpolation=cv2.INTER_NEAREST)
                overlay=np.zeros((280,280,4));overlay[:,:,:3]=[.96,.60,.19];overlay[:,:,3]=m*.48
                ax.imshow(overlay);ax.contour(m,levels=[.5],colors=['cyan'],linewidths=.6)
                ax.text(-.10,.5,'3CAD' if dataset=='3cad' else 'MVTec AD 2',transform=ax.transAxes,
                    rotation=90,ha='right',va='center',fontsize=7,weight='bold')
            else:
                im=ax.imshow(e0 if j==1 else e1,cmap='magma',vmin=0,vmax=upper,interpolation='nearest')
                ax.contour(a['sample_mask'],levels=[.5],colors=['#45CED5'],linewidths=.55)
                maps.append(ax)
            if row==0:ax.set_title(['Normal + mask','Single vMF','SDD-RD'][j],pad=4)
            ax.axis('off')
        p0=maps[0].get_position();p1=maps[1].get_position()
        cbax=fig.add_axes([p0.x0,p0.y0-.030,p1.x1-p0.x0,.013])
        bar=fig.colorbar(im,cax=cbax,orientation='horizontal');bar.set_ticks([0,upper/2,upper])
        bar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        bar.ax.tick_params(labelsize=5.5,pad=1,length=2)
        bar.set_label('Cosine reconstruction error',fontsize=6,labelpad=0)
        ax=fig.add_subplot(gs[row,4])
        vals=[reports[1][dataset]['vmf_single'],reports[1][dataset][PRIMARY],reports[0][dataset]['denoising_vmf']]
        x=np.array([v['clean_error'] for v in vals]);y=np.array([v['corrupted_region_error'] for v in vals])
        ax.plot(x,y,color=PURPLE,lw=1,marker='o',markersize=3)
        for i,lab in enumerate([r'$\rho=0$','0.25','1']):
            ax.annotate(lab,(x[i],y[i]),xytext=(4,4 if i!=2 else -9),textcoords='offset points',fontsize=6.2)
        ax.axvline(1.1*x[0],color=GRAY,lw=.7,ls='--')
        ax.set_xlim(x[0]*.975,max(x.max()*1.04,1.13*x[0]))
        ax.set_ylim(y.min()*.86,y.max()*1.16)
        ax.set_xlabel('Clean-target error',labelpad=1)
        ax.set_ylabel('Perturbed-region error',labelpad=2)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.locator_params(axis='x',nbins=3);ax.locator_params(axis='y',nbins=3)
        if row==0:ax.set_title('Normal-validation response',pad=4)
        ax.grid(alpha=.15,lw=.4)
        summary[dataset]={'normal_id':str(a['sample_id']),'clean_growth_percent':float((x[1]/x[0]-1)*100),
                         'perturbed_error_reduction_percent':float((1-y[1]/y[0])*100)}
    (OUT/'denoising_evidence.json').write_text(json.dumps(summary,indent=2))
    save(fig,'denoising_evidence')

def category_deltas():
    labels=[];diffs=[];colors=[];details=[]
    names={'Aluminum_Camera_Cover':'Al. camera cover','Aluminum_Ipad':'Al. iPad',
           'Aluminum_Middle_Frame':'Al. middle frame','Aluminum_New_Ipad':'Al. new iPad',
           'Aluminum_New_Middle_Frame':'Al. new middle frame','Aluminum_Pc':'Al. PC',
           'Copper_Stator':'Copper stator','Iron_Stator':'Iron stator','wallplugs':'Wall plugs'}
    for d in DATASETS:
        reports={v:[json.loads((ROOT/'results'/f'{d}_{v}_s{s}'/f'eval_{d}/metrics.json').read_text())['categories']
                    for s in SEEDS] for v in ['dinomaly',PRIMARY]}
        for cat in [x['category'] for x in reports['dinomaly'][0]]:
            x={v:np.array([[next(q[m] for q in rep if q['category']==cat)*100 for m in ['p_ap','aupro']]
                          for rep in reports[v]]) for v in reports}
            diff=x[PRIMARY]-x['dinomaly'];diffs.append(diff)
            labels.append(names.get(cat,cat.replace('_',' ').capitalize()))
            colors.append(BLUE if d=='3cad' else PURPLE)
            details.append({'dataset':d,'category':cat,'mean_delta':diff.mean(0).tolist(),'paired_seed_sd':diff.std(0,ddof=1).tolist()})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.05),sharey=True,gridspec_kw={'wspace':.13})
    ys=np.arange(16)[::-1]
    for k,ax in enumerate(axes):
        for y,ds,c in zip(ys,diffs,colors):
            mean=ds[:,k].mean();sd=ds[:,k].std(ddof=1)
            ax.hlines(y,0,mean,lw=.65,color=c,alpha=.5)
            ax.errorbar(mean,y,xerr=sd,fmt='o',color=c,markersize=3.4,capsize=1.6,lw=.7)
        ax.axvline(0,color='#69727A',lw=.8);ax.axhline(7.5,color='#BDC4CB',lw=.6)
        ax.set_xlabel('SDD-RD − Dinomaly (percentage points)')
        ax.set_title(['Pixel average precision','AUPRO'][k])
        ax.grid(axis='x',alpha=.15,lw=.5)
        ax.set_yticks(ys,labels if k==0 else [])
    axes[0].set_yticks(ys,labels)
    for k,ax in enumerate(axes):
        low=min(ds[:,k].mean()-ds[:,k].std(ddof=1) for ds in diffs)
        high=max(ds[:,k].mean()+ds[:,k].std(ddof=1) for ds in diffs)
        ax.set_xlim(np.floor(low)-.5,np.ceil(high)+.5)
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=.21,right=.985,top=.9,bottom=.12)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='3CAD'),
                        Line2D([],[],marker='o',ls='',color=PURPLE,label='MVTec AD 2')],
               loc='upper center',bbox_to_anchor=(.62,1.01),ncol=2,frameon=False)
    (OUT/'category_deltas.json').write_text(json.dumps(details,indent=2))
    save(fig,'category_comparison')

def native_map(raw,shape,size=280):
    x=cv2.resize(raw,(size,size),interpolation=cv2.INTER_LINEAR)
    x=cv2.GaussianBlur(x,(0,0),sigmaX=4.,borderType=cv2.BORDER_REFLECT)
    return cv2.resize(x,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)

def qualitative():
    deltas=json.loads((OUT/'category_deltas.json').read_text())
    specs=[]
    for d in DATASETS:
        selected=[x for x in deltas if x['dataset']==d]
        specs += [(d,max(selected,key=lambda x:x['mean_delta'][1])['category']),
                  (d,min(selected,key=lambda x:x['mean_delta'][1])['category'])]
    variants=['dinomaly','vmf_single',PRIMARY];display=['Dinomaly','Single vMF','SDD-RD']
    pred={}
    for d in DATASETS:
        for v in variants:
            f=ROOT/'results'/f'{d}_{v}_s17'/f'eval_{d}'
            pred[d,v]=(np.load(f/'predictions.npz')['maps'],json.loads((f/'rows.json').read_text()))
    fig=plt.figure(figsize=(7.16,4.45))
    gs=fig.add_gridspec(4,6,width_ratios=[1,1,1,1,1,1.25],wspace=.10,hspace=.31,
                       left=.02,right=.987,top=.91,bottom=.10)
    selection=[]
    short={'Aluminum_New_Middle_Frame':'3CAD · new middle frame','Iron_Stator':'3CAD · iron stator',
           'rice':'AD 2 · rice','vial':'AD 2 · vial'}
    for ri,(d,cat) in enumerate(specs):
        rows=pred[d,'dinomaly'][1]
        candidates=[]
        for i,r in enumerate(rows):
            if r['category']==cat and r['label']:
                mask=np.asarray(Image.open(r['mask']).convert('L'))>0
                candidates.append((float(mask.mean()),r['id'],i))
        candidates.sort();_,_,pos=candidates[len(candidates)//2]
        row=rows[pos];mask=np.asarray(Image.open(row['mask']).convert('L'))>0
        h,w=mask.shape
        image=np.asarray(Image.open(row['image']).convert('RGB').resize((w,h),Image.Resampling.BILINEAR))
        n,cc,stats,cent=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        bx,by,bw,bh,area=stats[component]
        side=int(max(bw,bh)*2.0);side=max(side,int(min(h,w)*.18));side=min(side,min(h,w))
        cx,cy=cent[component];x0=int(np.clip(cx-side/2,0,w-side));y0=int(np.clip(cy-side/2,0,h-side))
        x1=x0+side;y1=y0+side
        cropmask=mask[y0:y1,x0:x1]
        line=int(np.argmax(cropmask.sum(1)))
        aps=[];profiles=[]
        ax=fig.add_subplot(gs[ri,0]);ax.imshow(image)
        ax.add_patch(Rectangle((x0,y0),side,side,fill=False,edgecolor='#40CDD3',lw=1))
        ax.text(0,1.06,short.get(cat,cat),transform=ax.transAxes,fontsize=6.4,ha='left',va='bottom')
        ax.axis('off')
        ax=fig.add_subplot(gs[ri,1]);ax.imshow(image[y0:y1,x0:x1]);ax.contour(cropmask,levels=[.5],colors='cyan',linewidths=.5)
        ax.axhline(line,color='white',lw=.5,ls='--');ax.axis('off')
        if ri==0:ax.set_title('Defect crop',pad=14)
        for j,(v,label) in enumerate(zip(variants,display)):
            maps,vrows=pred[d,v];idx=next(i for i,x in enumerate(vrows) if x['id']==row['id'])
            assert (vrows[idx]['native_height'],vrows[idx]['native_width'])==(h,w)
            scores=native_map(maps[idx],mask.shape)
            ap=float(average_precision_score(mask.ravel(),scores.ravel()))*100;aps.append(ap)
            ranked=rankdata(scores.ravel(),method='average').reshape(scores.shape)/scores.size
            cropped=ranked[y0:y1,x0:x1]
            profiles.append(cropped[line])
            ax=fig.add_subplot(gs[ri,j+2])
            im=ax.imshow(cropped,cmap='magma',vmin=0,vmax=1)
            ax.contour(cropmask,levels=[.5],colors='cyan',linewidths=.45)
            ax.axhline(line,color='white',lw=.45,ls='--')
            ax.text(.5,-.05,f'AP {ap:.1f}',transform=ax.transAxes,ha='center',va='top',fontsize=6.3)
            if ri==0:ax.set_title(label,pad=14)
            ax.axis('off')
        ax=fig.add_subplot(gs[ri,5])
        x=np.linspace(0,1,side)
        ax.fill_between(x,0,1,where=cropmask[line],color='#D2D8DD',step='mid',zorder=0)
        for pr,c in zip(profiles,[BLUE,GREEN,PURPLE]):ax.plot(x,pr,color=c,lw=.75)
        ax.set(xlim=(0,1),ylim=(0,1.03),xticks=[0,.5,1],yticks=[0,1])
        ax.tick_params(pad=1,labelsize=5.5)
        if ri==0:ax.set_title('Cross-section',pad=14)
        if ri==3:ax.set_xlabel('Relative crop position',fontsize=6,labelpad=1)
        selection.append({'dataset':d,'category':cat,'id':row['id'],'native_crop':[x0,y0,x1,y1],
                          'profile_crop_row':line,'whole_image_pixel_ap':dict(zip(variants,aps)),
                          'category_selection':'largest positive / negative three-seed AUPRO difference',
                          'image_selection':'median anomalous mask area fraction, ties by identifier'})
    fig.legend(handles=[Line2D([],[],color=c,label=l,lw=1) for c,l in zip([BLUE,GREEN,PURPLE],display)],
               loc='lower center',bbox_to_anchor=(.58,-.006),ncol=3,frameon=False)
    cax=fig.add_axes([.045,.035,.18,.013]);bar=fig.colorbar(im,cax=cax,orientation='horizontal')
    bar.set_ticks([0,.5,1]);bar.ax.tick_params(labelsize=5.7,pad=1)
    bar.set_label('Within-image score percentile',fontsize=6,labelpad=1)
    (OUT/'qualitative_selection.json').write_text(json.dumps({'selection':selection,
        'color_scale':'within-image pixel-score ranks; shared 0-1 scale',
        'profile_shading':'ground-truth defect along the displayed cross-section',
        'AP':'Exact pixel average precision on the whole native-resolution image, seed 17.'},indent=2))
    save(fig,'qualitative')

if __name__=='__main__':
    denoising_evidence();category_deltas();qualitative()
    print('Denoising evidence, category effects, and native-crop model responses written.')
