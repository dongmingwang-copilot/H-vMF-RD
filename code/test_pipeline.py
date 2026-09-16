"""Verify that normal-validation decisions control denoising test access."""
import json
import pipeline


def test_partial_predictions_remain_pending(tmp_path):
    assert not pipeline.prediction_complete(tmp_path)
    (tmp_path/'predictions.npz').touch()
    assert not pipeline.prediction_complete(tmp_path)
    (tmp_path/'rows.json').write_text('[]')
    assert not pipeline.prediction_complete(tmp_path)
    (tmp_path/'prediction_metadata.json').write_text('{}')
    assert pipeline.prediction_complete(tmp_path)


def gate(root,name,decisions):
    folder = root/'research'
    folder.mkdir(exist_ok=True)
    (folder/name).write_text(json.dumps({'gate':decisions}))


def test_development_has_no_test_access_without_a_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    assert len(pipeline.development_jobs())==4
    assert not any(variant.startswith('denoising_') for _,variant,_ in pipeline.active_jobs())


def test_only_failed_objectives_receive_lower_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    gate(tmp_path,'denoising_validation.json',{'denoising_cosine':False,'denoising_vmf':True})
    development = pipeline.development_jobs()
    assert len(development)==6
    assert {(d,v,s) for d,v,s in development if v.endswith('_p25')}=={
        (dataset,'denoising_cosine_p25',17) for dataset in pipeline.DATASETS}
    repeats = {(d,v,s) for d,v,s in pipeline.active_jobs() if v.startswith('denoising_')}
    assert repeats=={(d,'denoising_vmf',s) for d in pipeline.DATASETS for s in [17,29,43]}


def test_predictions_include_all_passed_repeats_and_exclude_failed_models(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    gate(tmp_path,'denoising_validation.json',{'denoising_cosine':False,'denoising_vmf':True})
    gate(tmp_path,'denoising_p25_validation.json',{'denoising_cosine_p25':True})
    for dataset in pipeline.DATASETS:
        folder = tmp_path/'data'/dataset
        (folder/'features_280').mkdir(parents=True)
        (folder/'manifest.json').write_text('[]')
        (folder/'features_280/COMPLETE').touch()
    for dataset,variant,seed in set(pipeline.development_jobs()+pipeline.active_jobs()):
        run = tmp_path/'results'/f'{dataset}_{variant}_s{seed}'
        run.mkdir(parents=True)
        (run/'COMPLETE').touch()
    predictions = {(run,target,score) for run,target,score in pipeline.prediction_jobs() if '_denoising_' in run}
    expected = {(f'{source}_{variant}_s{seed}',target,'default') for source in pipeline.DATASETS
                for target in pipeline.DATASETS for variant in ['denoising_vmf','denoising_cosine_p25']
                for seed in [17,29,43]}
    assert predictions==expected
