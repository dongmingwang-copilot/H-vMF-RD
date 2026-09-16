"""Validate experiment provenance and export measured category statistics."""
import csv
import hashlib
import json
from pathlib import Path
import math

from pipeline import active_jobs, DATASETS, development_jobs, prediction_jobs, prediction_folder

ROOT = Path(__file__).resolve().parent.parent
METRICS = ['i_auroc','p_auroc','p_ap','aupro']


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    manifests = {dataset:json.loads((ROOT/'data'/dataset/'manifest.json').read_text()) for dataset in DATASETS}
    from baselines import selected_rows
    splits = {dataset:{split:selected_rows(dataset,split) for split in ['train','validation','test']} for dataset in DATASETS}
    pending, completed, measurements = [], [], []
    scheduled = set(active_jobs() + development_jobs())
    measured = set()
    for path in (ROOT/'results').glob('*/config.json'):
        config = json.loads(path.read_text())
        if (path.parent/'COMPLETE').exists():
            measured.add((config['dataset'],config['variant'],config['seed']))
    expected = {}
    for name,target,score in prediction_jobs():
        expected.setdefault(name,set()).add((target,score))
    for source,variant,seed in sorted(scheduled | measured):
        name = f'{source}_{variant}_s{seed}'
        run = ROOT/'results'/name
        if not (run/'COMPLETE').exists():
            pending.append({'run':name,'stage':'training'})
            continue
        config = json.loads((run/'config.json').read_text())
        assert config['dataset'] == source and config['variant'] == variant and config['seed'] == seed
        if variant != 'patchcore':
            assert int((run/'COMPLETE').read_text()) == config['steps'] == 10000
        recorded = json.loads((run/'split_ids.json').read_text())
        assert recorded['train'] == [row['id'] for row in splits[source]['train']]
        if 'validation' in recorded:
            assert recorded['validation'] == [row['id'] for row in splits[source]['validation']]
        checkpoint = run/('memory.npy' if variant=='patchcore' else 'model.pt')
        checkpoint_hash = digest(checkpoint)
        evaluations = set(expected.get(name,()))
        for path in run.glob('eval_*/prediction_metadata.json'):
            prediction = json.loads(path.read_text())
            evaluations.add((prediction['target'],prediction.get('score','default')))
        for target,score in sorted(evaluations):
            folder = prediction_folder(name,target,score)
            if not (folder/'metrics.json').exists():
                pending.append({'run':name,'target':target,'score':score,'stage':'evaluation'})
                continue
            report = json.loads((folder/'metrics.json').read_text())
            prediction = json.loads((folder/'prediction_metadata.json').read_text())
            rows = json.loads((folder/'rows.json').read_text())
            assert report['run']==name and report['target']==prediction['target']==target
            assert prediction.get('score','default')==score
            if 'feature_manifest_sha256' in prediction:
                assert prediction['feature_manifest_sha256']==digest(ROOT/'data'/target/'manifest.json')
            assert prediction.get('checkpoint_sha256',prediction.get('memory_sha256'))==checkpoint_hash
            assert [r['id'] for r in rows]==[r['id'] for r in splits[target]['test']]
            categories = sorted({row['category'] for row in splits[target]['test']})
            assert sorted(row['category'] for row in report['categories'])==categories
            for row in report['categories']:
                selected = [r for r in rows if r['category']==row['category']]
                assert row['images']==len(selected)
                assert row['anomalous_images']==sum(r['label'] for r in selected)
                assert row['positive_pixels']+row['background_pixels']==sum(r['native_width']*r['native_height'] for r in selected)
                assert row['aupro_limit']==(.05 if target=='mvtec_ad2' else .30)
                assert all(math.isfinite(row[key]) and 0<=row[key]<=1 for key in METRICS)
                measurements.append({'source':source,'target':target,'variant':variant,'seed':seed,'score':score,
                                     'category':row['category'],**{key:row[key] for key in METRICS}})
            for key in METRICS:
                assert abs(report['mean'][key]-sum(r[key] for r in report['categories'])/len(categories))<1e-12
            measurements.append({'source':source,'target':target,'variant':variant,'seed':seed,'score':score,
                                 'category':'macro_average',**report['mean']})
            completed.append({'run':name,'target':target,'score':score,'checkpoint_sha256':checkpoint_hash,
                              'metrics_sha256':digest(folder/'metrics.json'),'metrics':report['mean']})
    (ROOT/'research').mkdir(exist_ok=True)
    audit = {'completed_evaluations':len(completed),'completed':completed,'pending':pending,
             'manifests':{dataset:{'images':len(rows),'sha256':digest(ROOT/'data'/dataset/'manifest.json')}
                          for dataset,rows in manifests.items()}}
    (ROOT/'research/experiment_audit.json').write_text(json.dumps(audit,indent=2))
    with (ROOT/'results/measurements.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['source','target','variant','seed','score','category',*METRICS])
        writer.writeheader()
        writer.writerows(measurements)
    print(json.dumps({'completed_evaluations':len(completed),'pending_stages':len(pending),'checks':'passed'}))


if __name__=='__main__':
    main()
