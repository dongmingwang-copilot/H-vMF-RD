"""Category-resolved CFRG comparison and fixed complementary component cases."""
from pathlib import Path
import json,hashlib
import numpy as np,cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import average_precision_score
from common import ROOT,WORK
OUT=WORK/"submission_case_study"/"presentation";F=OUT/"figures";G=OUT/"generated"
F.mkdir(parents=True,exist_ok=True);G.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"axes.linewidth":.5})
V=["normal_continue","whole_denoise","spatial_target"];seeds=[17,29,43]
def metric(v,s=17):return json.loads((WORK/f"runs/3cad_{v}_s{s}/eval_3cad_12000/metrics.json").read_text())
D={v:[metric(v,s) for s in seeds] for v in V};C=metric("cfrg448")
cats=[r["category"] for r in C["categories"]]
labels=dict(zip(cats,["Camera cover","Tablet housing","Middle frame","New tablet housing","New middle frame","PC housing","Copper stator","Iron stator"]))
def vals(v,c,key):return np.array([next(z[key] for z in a["categories"] if z["category"]==c) for a in D[v]])*100
lines=[];data=[]
for c in cats:
    cf=next(z for z in C["categories"] if z["category"]==c);ref=next(z for z in D[V[0]][0]["categories"] if z["category"]==c)
    pp=100*ref["positive_pixels"]/(ref["positive_pixels"]+ref["background_pixels"])
    cells=[labels[c],f'{ref["images"]:,}',f'{pp:.3f}']
    row={"category":c,"label":labels[c],"test_images":ref["images"],"foreground_percent":pp}
    for k in ["p_ap","aupro"]:
        a=vals("spatial_target",c,k);cv=cf[k]*100
        def bf(t,yes):return r"\textbf{"+t+"}" if yes else t
        cells += [f'{bf(f"{a.mean():.2f}",a.mean()>cv)} $\\pm$ {a.std(ddof=1):.2f}',bf(f"{cv:.2f}",cv>a.mean())]
        row[k]={"ssrd_mean":float(a.mean()),"ssrd_sd":float(a.std(ddof=1)),"cfrg":cv,"delta":float(a.mean()-cv)}
    lines.append(" & ".join(cells)+r"\\");data.append(row)
table=r"""\begin{table*}[!t]
\centering
\caption{Component-resolved localization on the complete 3CAD test set (\%). Foreground is the fraction of annotated defect pixels among all test pixels, including normal images. SS-RD reports three-seed means and sample standard deviations; CFRG uses seed 17. Bold identifies the higher point estimate in each pair.}
\label{tab:component_cfrg}
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{P-AP} & \multicolumn{2}{c}{AUPRO}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Component & Test images & Foreground & SS-RD & CFRG & SS-RD & CFRG\\
\midrule
"""+ "\n".join(lines)+r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
(G/"component_cfrg.tex").write_text(table)
(F/"component_cfrg_data.json").write_text(json.dumps(data,indent=2))
# Four categories complement the four cases in the existing within-backbone figure.
casecats=["Aluminum_Ipad","Aluminum_New_Ipad","Aluminum_New_Middle_Frame","Copper_Stator"]
paths={v:WORK/f"runs/3cad_{v}_s17/eval_3cad_12000" for v in ["normal_continue","spatial_target","cfrg448"]}
rows=json.loads((paths["spatial_target"]/"rows.json").read_text())
for q in paths.values():assert [r["id"] for r in json.loads((q/"rows.json").read_text())]==[r["id"] for r in rows]
maps={v:np.load(q/"predictions.npz")["maps"] for v,q in paths.items() if v!="cfrg448"}
maps["cfrg448"]=np.load(paths["cfrg448"]/"maps.npy",mmap_mode="r")
fig=plt.figure(figsize=(7.1,5.2))
gs=fig.add_gridspec(4,5,width_ratios=[1.05,1,1,1,1],hspace=.45,wspace=.105)
fig.subplots_adjust(left=.035,right=.98,top=.95,bottom=.07)
entries=[]
for j,cat in enumerate(casecats):
    ids=[i for i,r in enumerate(rows) if r["category"]==cat and r["label"]]
    i=min(ids,key=lambda i:hashlib.sha256(("localization-audit:"+rows[i]["id"]).encode()).hexdigest())
    row=rows[i];mask=np.asarray(Image.open(row["mask"]).convert("L"))>0;h,w=mask.shape
    rgb=cv2.resize(np.asarray(Image.open(row["image"]).convert("RGB")),(w,h))
    _,comp,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    largest=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());xx,yy,bw,bh=stats[largest,:4];cx,cy=centroids[largest]
    side=int(max(96,min(min(h,w),max(2*max(bw,bh),min(h,w)//4))))
    x0=int(max(0,min(cx-side/2,w-side)));y0=int(max(0,min(cy-side/2,h-side)))
    crop=(slice(y0,y0+side),slice(x0,x0+side));cm=mask[crop]
    ms=[];aps=[]
    for v in ["normal_continue","spatial_target","cfrg448"]:
        raw=maps[v][i]
        if v!="cfrg448":raw=cv2.GaussianBlur(cv2.resize(raw,(448,448)),(0,0),4.,borderType=cv2.BORDER_REFLECT)
        m=cv2.resize(raw,(w,h));ms.append(m)
        aps.append(float(average_precision_score(mask.ravel(),m.ravel()))*100)
    lows=[min(float(m.min()) for m in ms[:2])]*2+[float(ms[2].min())]
    highs=[max(float(m.max()) for m in ms[:2])]*2+[float(ms[2].max())]
    axs=[]
    for col in range(5):
        ax=fig.add_subplot(gs[j,col]);axs.append(ax);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        if col==0:
            ax.imshow(rgb);ax.add_patch(Rectangle((x0,y0),side,side,fill=False,ec="#D88A32",lw=.8))
            ax.set_ylabel(labels[cat].replace("New ","New\n").replace(" housing","\nhousing"),labelpad=3,fontsize=8)
        else:
            ax.imshow(rgb[crop])
            if col>=2:
                k=col-2
                ax.imshow(ms[k][crop],cmap="magma",vmin=lows[k],vmax=highs[k],alpha=.78)
                ax.text(.5,-.045,f"AP {aps[k]:.1f}%",transform=ax.transAxes,ha="center",va="top",fontsize=7.6)
            ax.contour(cm,levels=[.5],colors=["#26C9AF"],linewidths=.65)
        if j==0:ax.set_title(["Image / crop","Annotation","Normal continuation","SS-RD","CFRG"][col],fontsize=8,pad=7)
    for idxs,lo,hi in [([2,3],lows[0],highs[0]),([4],lows[2],highs[2])]:
        a=axs[idxs[0]].get_position();b=axs[idxs[-1]].get_position()
        cax=fig.add_axes([a.x0,a.y0-.036,b.x1-a.x0,.007])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(lo,hi),cmap="magma"),cax=cax,orientation="horizontal")
        cb.set_ticks([lo,hi]);cb.ax.set_xticklabels([f"{lo:.3f}",f"{hi:.3f}"]);cb.ax.tick_params(labelsize=7,length=1,pad=1)
    entries.append({"id":row["id"],"category":cat,"crop":[x0,y0,side,side],"full_image_pixel_ap":dict(zip(["normal_continue","ssrd","cfrg"],aps)),
       "raw_limits":{"matched":[lows[0],highs[0]],"cfrg":[lows[2],highs[2]]}})
for suffix in ["pdf","png"]:fig.savefig(F/f"component_cases.{suffix}",bbox_inches="tight",pad_inches=.025,dpi=220)
plt.close(fig)
(F/"component_cases_data.json").write_text(json.dumps({"selection":"minimum SHA256(localization-audit:id); remaining four 3CAD categories","entries":entries,
 "display":"normal and SS-RD share full-image raw range; CFRG uses separate raw scale because its score adds segmentation and three residuals; no pixel values are altered"},indent=2))
print(json.dumps(entries),flush=True)
