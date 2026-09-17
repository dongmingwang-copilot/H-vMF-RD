"""Run the fixed experiments with restartable outputs and no scheduler."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;CODE=ROOT/"code/ssrd"
def call(name,*args):
    subprocess.run([sys.executable,"-u",str(CODE/name),*map(str,args)],cwd=ROOT,check=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--datasets",default="3cad,mvtec_ad2")
    ap.add_argument("--seeds",default="17,29,43")
    for option in ["external","transfer","resolution","timing"]:ap.add_argument("--skip-"+option,action="store_true")
    a=ap.parse_args();datasets=a.datasets.split(",");seeds=[int(s) for s in a.seeds.split(",")]
    assert set(datasets)<={"3cad","mvtec_ad2"} and set(seeds)<={17,29,43}
    for seed in seeds:
        for ds in datasets:
            call("train_v2.py","--dataset",ds,"--variant","plain","--tag","stable","--optimizer","stable",
                 "--seed",seed,"--steps",10000,"--stop",10000)
            if seed==17:
                for v in ["normal_continue","whole_denoise","spatial_target"]:call("spatial_target.py","--dataset",ds,"--variant",v)
            else:call("refine_paired.py","--dataset",ds,"--seed",seed)
            call("evaluate_v2.py","--source",ds,"--target",ds,"--seed",seed,"--step",12000,
                 "--variants","normal_continue,whole_denoise,spatial_target")
    if not a.skip_resolution:
        assert 17 in seeds,"Resolution experiment requires seed17"
        for ds in datasets:
            call("highres_control.py","--dataset",ds)
            variants=["spatial_highres","whole_highres"] if ds=="3cad" else ["spatial_highres"]
            for v in variants:call("spatial_highres.py","--dataset",ds,"--variant",v)
            call("evaluate_conditioned.py","--dataset",ds,"--variants",",".join(["highres_continue"]+variants))
    if not a.skip_external:
        for ds in datasets:
            for v in ["rd","rdpp"]:
                call("train_cnn.py","--dataset",ds,"--variant",v)
                call("evaluate_cnn.py","--source",ds,"--target",ds,"--variant",v)
            call("train_hfsa.py","--dataset",ds)
            call("evaluate_hfsa.py","--source",ds,"--target",ds)
        if "3cad" in datasets:
            call("train_cfrg.py")
            call("evaluate_cfrg.py")
    if not a.skip_transfer:
        assert set(datasets)=={"3cad","mvtec_ad2"} and 17 in seeds,"Transfer requires both datasets and seed17"
        for source,target in [("3cad","mvtec_ad2"),("mvtec_ad2","3cad")]:
            call("evaluate_v2.py","--source",source,"--target",target,"--step",12000,
                 "--variants","normal_continue,whole_denoise,spatial_target")
    if not a.skip_timing:
        call("benchmark_inference.py")
        if "3cad" in datasets and not a.skip_external:
            call("benchmark_inference.py","--append-cfrg")
if __name__=="__main__":main()
