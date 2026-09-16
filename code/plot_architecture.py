"""Publication architecture: an overview and two visual module expansions."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper/figures'
OUT.mkdir(parents=True,exist_ok=True)
C={'t':'#4477AA','d':'#228833','p':'#AA3377','n':'#BB7733','g':'#46515B'}
F={'t':'#EDF3F9','d':'#EDF6F0','p':'#FAF0F6','n':'#FCF4EB','g':'#F6F7F8'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.3,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320})

def architecture():
    fig=plt.figure(figsize=(7.16,4.30))
    ax=fig.add_axes([.005,.005,.99,.99]);ax.set(xlim=(0,100),ylim=(0,60));ax.axis('off')
    def text(x,y,s,fs=7.3,ha='center',color='#25313A',weight='normal',va='center'):
        return ax.text(x,y,s,fontsize=fs,ha=ha,va=va,color=color,weight=weight,zorder=7)
    def box(x,y,w,h,k='g'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.65',
            facecolor=F[k],edgecolor=C[k],lw=.75,zorder=1))
    def arr(points,k='g',dash=False):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=C[k],lw=.8,ls='--' if dash else '-',zorder=3)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=6.5,
            color=C[k],lw=.8,linestyle='--' if dash else '-',zorder=3))
    def grid(x,y,w,h,k='t',mask=False,n=5):
        for i in [2,1,0]:
            ax.add_patch(plt.Rectangle((x+i*.3,y+i*.28),w,h,fc='white',ec=C[k],lw=.6,zorder=3))
        for i in range(n):
            for j in range(n):
                selected=mask and ((i<max(1,n//4) and j<max(1,n//4)) or (n//2<=i<n//2+max(1,n//4) and n//2<=j<n//2+max(1,n//4)))
                cc=C['n'] if selected else (C['t'] if k=='n' else C[k])
                ax.add_patch(plt.Rectangle((x+.25+j*(w-.5)/n,y+.25+i*(h-.5)/n),
                    (w-.5)/n-.13,(h-.5)/n-.13,fc=cc,alpha=.25 if not selected else .8,ec='none',zorder=4))
    def stage(x,y,w,h,label,k):
        box(x,y,w,h,k);text(x+w/2,y+h/2,label,6.9)
    # Panel (a): a compact and correctly connected computational overview.
    text(1,58.3,'(a)  Reverse distillation with directional reconstruction',8.2,'left',weight='bold')
    rows=json.loads((ROOT/'data/3cad/manifest.json').read_text())
    sample=next(r for r in rows if r['split'] in {'test','test_public'} and r['label'])
    im=np.asarray(Image.open(sample['image']).convert('RGB').resize((96,96)))
    ax.imshow(im,extent=(1,9,44.5,52.5),zorder=2)
    text(5,42.5,'Input image',6.8)
    box(12,43.8,17,10,'t')
    text(20.5,52,'Frozen DINOv2',7.1)
    for j in range(8):
        ax.add_patch(plt.Rectangle((13.3+j*1.85,47.3),1.35,2.8,fc=C['t'],alpha=.45 if j<4 else .8,ec='white',lw=.4))
    text(16.5,45.6,'2–5',6.5);text(24,45.6,'6–9',6.5)
    grid(32,48.2,5.3,4.2);grid(32,41.6,5.3,4.2)
    text(40,50.3,r'$T_1$',7.5);text(40,43.7,r'$T_2$',7.5)
    stage(43,43.8,9,10,'SDD\ntrain only','n')
    stage(55,43.8,10,10,'Mean\n+ MLP','d')
    box(68,43.8,17,10,'d')
    for j in range(8):
        ax.add_patch(plt.Rectangle((69+j*1.85,47),1.4,3.7,fc=C['d'],alpha=.42 if j<4 else .75,ec='white',lw=.4))
    text(76.5,52,'Reverse decoder',6.9)
    text(76.5,45.4,'8 blocks',6.7)
    stage(88,43.8,11,10,'vMF heads\n+ NLL','p')
    for a,b in [(9,12),(29,31.5),(40.7,43),(52,55),(65,68),(85,88)]: arr([(a,48.8),(b,48.8)])
    arr([(41.5,48.8),(41.5,56),(60,56),(60,53.8)],'g')
    text(51.5,57,'inference bypass',6.3)
    # Clean targets enter the likelihood, not the corrupted feature branch.
    arr([(35,41),(35,39),(93.5,39),(93.5,43.8)],'t',True)
    text(60,40.1,'clean teacher targets',6.6,color=C['t'])
    text(8,38.9,'Solid: forward path',6.3,ha='left')
    text(8,37.2,'Dashed: supervision',6.3,ha='left',color=C['t'])
    # Module expansion (b).
    box(.6,1,48.2,34.7,'n')
    text(2.2,33.5,'(b)  Structured directional denoising',7.8,'left',weight='bold')
    grid(3,24,6.5,6);grid(3,14,6.5,6)
    text(6.2,22.3,'Recipient',6.7);text(6.2,12.3,'Rolled donor',6.7)
    arr([(10,27),(13,27)],'n');arr([(10,17),(13,17),(13,24),(15,24)],'n')
    # Directions on a unit circle, with the short-arc midpoint explicit.
    cx,cy,rad=21,24.3,5.9
    ax.add_patch(Circle((cx,cy),rad,fc='white',ec='#DBB58C',lw=.6,zorder=2))
    for angle,color,label in [(18,C['t'],r'$y$'),(72,C['t'],r'$y^{\prime}$'),(45,C['n'],r'$m$')]:
        end=(cx+rad*.83*np.cos(np.deg2rad(angle)),cy+rad*.83*np.sin(np.deg2rad(angle)))
        arr([(cx,cy),end], 'n' if angle==45 else 't')
        text(end[0]+.8,end[1]+.9,label,7,color=color)
    ax.add_patch(Arc((cx,cy),rad*1.65,rad*1.65,theta1=18,theta2=72,color=C['n'],lw=1.3,zorder=4))
    text(21,16.1,r'$m=(y+y^{\prime})/\|y+y^{\prime}\|$',6.8)
    arr([(27.5,24.3),(30,24.3)],'n')
    grid(31,22,7,7,'n',True,n=20)
    text(34.5,20.2,'Shared mask',6.7)
    text(34.5,17.9,r'$2\times(5\times5)$',6.6)
    arr([(38.8,25.4),(41,25.4)],'n')
    grid(41.5,23,5.2,5.2,'t',True)
    text(44,21.4,r'$\widetilde T_h$',7.4,color=C['n'])
    # Lower strip fills the space with the actual algebraic block and constraints.
    stage(3,4.1,42.5,6.5,r'$\widetilde T_h=r_h[(1-M)y_h+M m_h]$','n')
    text(24.5,12.6,r'$\rho=0.25$  ·  shared across $h=1,2$',6.8)
    text(24.5,2.4,'Token norms and prefix tokens are retained',6.4,color=C['n'])
    # Module expansion (c): the statistical head, targets, and distinct reductions.
    box(50,1,49.4,34.7,'p')
    text(51.6,33.5,'(c)  Directional head and anomaly scoring',7.8,'left',weight='bold')
    grid(52,22.5,6,6,'d')
    text(55,20.5,r'$R_h$',7.3,color=C['d'])
    stage(61,25.1,15,5,'Residual + norm','p')
    stage(61,16.8,15,5,'Pool + LN + linear','p')
    arr([(58.7,26.5),(61,27.6)],'d')
    arr([(59,25.5),(59,19.3),(61,19.3)],'d')
    arr([(76,27.6),(79,27.6)],'p');arr([(76,19.3),(79,19.3)],'p')
    # Mean arrow and concentration density curves.
    ax.add_patch(Circle((82,27.4),2.4,fc='white',ec=C['p'],lw=.7,zorder=2))
    arr([(82,27.4),(83.4,29)],'p');text(87,27.4,r'$\mu_{h,p}$',7.5,color=C['p'])
    xx=np.linspace(79.5,85,60)
    ax.plot(xx,17.4+2.9*np.exp(-((xx-82.3)/.9)**2),color=C['p'],lw=1)
    ax.plot(xx,17.4+1.4*np.exp(-((xx-82.3)/1.7)**2),color=C['p'],lw=.6,ls='--')
    text(88.4,19.3,r'$\kappa_h$',7.5,color=C['p'])
    arr([(91,27.4),(96,27.4),(96,15),(88,15),(88,13)],'p')
    arr([(91,19.3),(96,19.3)],'p')
    stage(52,7,39,6,r'$\ell_{h,p}=[B_d(\kappa_h)-\kappa_h\mu_{h,p}^{\top}y_{h,p}]/d$','p')
    text(52,15,'Clean target',6.5,ha='left',color=C['t'])
    arr([(57,14.1),(57,13)],'t',True)
    text(72,4.6,r'$A(p)=\frac{1}{2}\sum_h\ell_{h,p}$',7.4)
    text(72,2.5,'Resize + smooth; image score = maximum',6.2)
    arr([(91,8.8),(94,8.8)],'p')
    yy,xx=np.mgrid[:16,:12];heat=np.exp(-((xx-7)**2+(yy-9)**2)/6)
    ax.imshow(heat,cmap='magma',extent=(94,98.4,5.5,11.4),zorder=2)
    text(96.2,4.2,'Map',6.4)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'architecture.{ext}',facecolor='white')
    plt.close(fig)

def corruption():
    import torch
    from torch.nn import functional as F
    from train import CachedFeatures
    from model import directional_patch_corruption
    torch.set_num_threads(2)
    data = CachedFeatures('3cad',280,'train')
    positions = [0,1]
    features = torch.stack([data[i][0] for i in positions]).float()
    changed = directional_patch_corruption(features,generator=torch.Generator().manual_seed(17))
    cosine = F.cosine_similarity(features[0,0,5:],changed[0,0,5:],dim=-1).clamp(-1,1)
    angle = torch.rad2deg(torch.acos(cosine)).reshape(20,20).numpy()
    difference = (changed[0,0,5:]-features[0,0,5:]).abs().sum(-1).reshape(20,20).numpy()
    angle[difference<1e-5] = 0
    rows = [data.rows[data.indices[i]] for i in positions]
    fig,axes = plt.subplots(1,3,figsize=(3.5,1.7),layout='constrained')
    for ax,row,label in zip(axes[:2],rows,['Recipient','Normal donor']):
        ax.imshow(Image.open(row['image']).convert('RGB'))
        ax.set_title(label,fontsize=7,pad=4)
        ax.axis('off')
    im = axes[2].imshow(angle,cmap='viridis',vmin=0,vmax=90,interpolation='nearest')
    axes[2].set_title('Direction change',fontsize=7,pad=4)
    axes[2].set(xticks=[],yticks=[])
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.075,pad=.03,shrink=.65)
    bar.set_ticks([0,30,60,90])
    bar.ax.tick_params(labelsize=6,pad=1)
    bar.set_label('Angular change (degrees)',fontsize=7,labelpad=1)
    error = (features.norm(dim=-1)-changed.norm(dim=-1)).abs().max().item()
    (OUT/'corruption_provenance.json').write_text(json.dumps({'dataset':'3cad','split':'train',
        'ids':[row['id'] for row in rows],'generator_seed':17,'max_norm_error':error},indent=2))
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'corruption.{ext}',bbox_inches='tight',pad_inches=.035,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    architecture()
    corruption()
    print('Architecture written in PDF/SVG/PNG')
