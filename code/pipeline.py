"""Single-GPU experiment queue with concurrent CPU metric evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import time
from filelock import FileLock

ROOT=Path(__file__).resolve().parent.parent
PYTHON=sys.executable
DATASETS=['3cad','mvtec_ad2']
PRIMARY=[('vmf_fixed',17),('dinomaly',17),('coupled_mixture',17),('vmf_single',17),('vmf_mixture',17),('context_mixture',17),
         ('dinomaly',29),('coupled_mixture',29),('dinomaly',43),('coupled_mixture',43)]
JOBS=[(dataset,variant,17) for variant in ['patchcore','rd','rdpp'] for dataset in DATASETS]
JOBS += [(dataset,variant,seed) for variant,seed in PRIMARY for dataset in DATASETS]
DEVELOPMENT_JOBS = [(dataset,variant,17) for variant in ['denoising_cosine','denoising_vmf'] for dataset in DATASETS]


def development_jobs():
    jobs = list(DEVELOPMENT_JOBS)
    path = ROOT/'research/denoising_validation.json'
    if path.exists():
        gate = json.loads(path.read_text())['gate']
        jobs += [(dataset,variant+'_p25',17) for variant,passed in gate.items() if not passed for dataset in DATASETS]
    return jobs


def active_jobs():
    jobs = list(JOBS)
    for name in ['denoising_validation.json','denoising_p25_validation.json']:
        gate_path = ROOT/'research'/name
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())['gate']
            jobs += [(dataset,variant,seed) for variant,passed in gate.items() if passed
                     for seed in [17,29,43] for dataset in DATASETS]
    return jobs


def execute(script,arguments,name,tick=None):
    path=ROOT/'logs'/f'{name}.log'
    with path.open('a') as log:
        command=[PYTHON,'-u',str(ROOT/'code'/script),*arguments]
        print('START',name,flush=True)
        with FileLock(str(ROOT/'logs/gpu.lock')):
            result=subprocess.Popen(command,cwd=ROOT/'code',stdout=log,stderr=log)
            try:
                while result.poll() is None:
                    if tick is not None:
                        tick()
                    time.sleep(5)
            except BaseException:
                result.terminate()
                result.wait()
                raise
    if result.returncode:
        raise RuntimeError(f'{name} failed: {path}')
    print('FINISH',name,flush=True)


def ready(dataset):
    return (ROOT/'data'/dataset/'manifest.json').exists()


def cached(dataset):
    return (ROOT/'data'/dataset/'features_280/COMPLETE').exists()


def evaluation_jobs():
    for source,variant,seed in active_jobs():
        name=f'{source}_{variant}_s{seed}'
        if not (ROOT/'results'/name/'COMPLETE').exists():
            continue
        targets=[source]
        if variant not in ['vmf_fixed','vmf_single','vmf_mixture','context_mixture']:
            targets += [d for d in DATASETS if d!=source]
        for target in targets:
            if ready(target) and (variant in ['rd','rdpp','patchcore'] or cached(target)):
                yield name,target


def prediction_jobs():
    for name,target in evaluation_jobs():
        yield name,target,'default'
    for source in DATASETS:
        name=f'{source}_vmf_single_s17'
        if (ROOT/'results'/name/'COMPLETE').exists():
            for target in DATASETS:
                if cached(target):
                    for score in ['angular_standardized','angular_deviation']:
                        yield name,target,score


def prediction_folder(name,target,score):
    suffix='' if score=='default' else '_'+score
    return ROOT/'results'/name/f'eval_{target}{suffix}'


def prediction_complete(folder):
    return all((folder/name).is_file() for name in ['predictions.npz','rows.json','prediction_metadata.json'])


def main():
    for folder in ['logs','research','results']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    metric_processes={}
    def update_metrics(excluded=None):
        for key,(process,log) in list(metric_processes.items()):
            if process.poll() is not None:
                log.close()
                del metric_processes[key]
                if process.returncode:
                    raise RuntimeError(f'Metric evaluation failed: {key}')
        for name,target,score in prediction_jobs():
            output=prediction_folder(name,target,score)
            key=name+'_'+target+'_'+score
            if key==excluded:
                continue
            if prediction_complete(output) and not (output/'metrics.json').exists() and key not in metric_processes and len(metric_processes)<2:
                log=(ROOT/'logs'/f'metrics_{key}.log').open('a')
                process=subprocess.Popen([PYTHON,'-u',str(ROOT/'code/evaluate.py'),'--run',name,
                    '--target',target,'--score',score,'--metrics-only','--workers','4'],stdout=log,stderr=log,cwd=ROOT/'code')
                metric_processes[key]=(process,log)
    while True:
        update_metrics()
        for dataset in DATASETS:
            if ready(dataset) and not cached(dataset):
                execute('cache_features.py',[dataset],'cache_'+dataset,tick=update_metrics)
        development_complete=all((ROOT/'results'/f'{d}_{v}_s{s}'/'COMPLETE').exists() for d,v,s in DEVELOPMENT_JOBS)
        if development_complete and not (ROOT/'research/denoising_validation.json').exists():
            execute('validate_denoising.py',[],'validate_denoising',tick=update_metrics)
        development = development_jobs()
        secondary = [(d,v,s) for d,v,s in development if v.endswith('_p25')]
        if secondary and all((ROOT/'results'/f'{d}_{v}_s{s}/COMPLETE').exists() for d,v,s in secondary):
            if not (ROOT/'research/denoising_p25_validation.json').exists():
                execute('validate_denoising.py',['--p25'],'validate_denoising_p25',tick=update_metrics)
        pending_prediction=next(((n,t,s) for n,t,s in prediction_jobs() if not prediction_complete(prediction_folder(n,t,s))),None)
        if pending_prediction:
            name,target,score=pending_prediction
            key=name+'_'+target+'_'+score
            execute('evaluate.py',['--run',name,'--target',target,'--score',score,'--predict-only'],f'predict_{name}_{target}_{score}',tick=lambda:update_metrics(excluded=key))
            continue
        jobs = active_jobs()
        pending=next(((d,v,s) for d,v,s in development+jobs if cached(d) and not (ROOT/'results'/f'{d}_{v}_s{s}'/'COMPLETE').exists()),None)
        if pending:
            dataset,variant,seed=pending
            name=f'{dataset}_{variant}_s{seed}'
            if variant=='patchcore':
                execute('patchcore_runner.py',['--dataset',dataset,'--seed',str(seed)],name,tick=update_metrics)
            else:
                script='baselines.py' if variant in ['rd','rdpp'] else 'train.py'
                arguments=['--dataset',dataset,'--variant',variant,'--seed',str(seed),'--steps','10000']
                execute(script,arguments,name,tick=update_metrics)
            continue
        all_trained=all((ROOT/'results'/f'{d}_{v}_s{s}'/'COMPLETE').exists() for d,v,s in development+jobs)
        all_evaluated=all((prediction_folder(n,t,s)/'metrics.json').exists() for n,t,s in prediction_jobs())
        unique_jobs = set(development+jobs)
        state={'timestamp':time.time(),'trained':sum((ROOT/'results'/f'{d}_{v}_s{s}'/'COMPLETE').exists() for d,v,s in unique_jobs),
               'total_jobs':len(unique_jobs),'active_metric_jobs':list(metric_processes),'datasets_ready':[d for d in DATASETS if ready(d)]}
        (ROOT/'research/pipeline_state.json').write_text(json.dumps(state,indent=2))
        if all_trained and all_evaluated and not metric_processes:
            (ROOT/'results/EXPERIMENTS_COMPLETE').write_text(str(time.time()))
            print('All comparison, ablation, and transfer experiments complete',flush=True)
            break
        time.sleep(20)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        (ROOT/'research/pipeline_error.txt').write_text(str(error))
        raise
