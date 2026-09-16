import cv2
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from metrics import PixelMetrics, partial_area, VectorizedStatCurve
from adeval.mem_effic import _AccumulateStatCurve
from train import FixedStepBatches


@pytest.mark.parametrize('kind', ['perfect', 'reversed', 'constant', 'random'])
def test_pixel_metrics_against_exact_rank_metrics(kind):
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    mask[10:25, 10:25] = 1
    score = {'perfect': mask.astype(float), 'reversed': 1.-mask,
             'constant': np.ones_like(mask, dtype=float),
             'random': np.random.default_rng(9).random(mask.shape)}[kind]
    accumulator = PixelMetrics(float(score.min()), float(score.max()))
    accumulator.add(score, mask)
    output = accumulator.summary(0.3)
    assert output['p_auroc'] == pytest.approx(roc_auc_score(mask.ravel(), score.ravel()), abs=1e-5)
    assert output['p_ap'] == pytest.approx(average_precision_score(mask.ravel(), score.ravel()), abs=1e-5)
    if kind == 'perfect':
        assert output['aupro'] == pytest.approx(1.)
    if kind == 'constant':
        assert output['aupro'] == pytest.approx(0.15)
    assert output['defect_regions'] == 2


def test_regions_are_equally_weighted_across_images():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:15, 5:15] = 1
    score = np.full(mask.shape, 0.5)
    score[mask > 0] = 0.
    score[1:3, 1:3] = 1.
    acc = PixelMetrics(0, 1)
    acc.add(score, mask)
    # The small four-pixel region has half the PRO weight.
    assert acc.summary(0.3)['aupro'] == pytest.approx(0.5)


def test_partial_area_interpolates_endpoint():
    assert partial_area([0., 0.1, 0.5, 1.], [0., 0.4, 0.8, 1.], 0.3) == pytest.approx(0.4)
    assert partial_area([0., 0., 0.5, 1.], [0., 0.7, 0.8, 1.], 0.05) == pytest.approx(0.705)


def test_sampler_resume_reproduces_batches():
    full = list(FixedStepBatches(51, 8, 17, 0, 25))
    resumed = list(FixedStepBatches(51, 8, 17, 11, 25))
    assert full[11:] == resumed
    assert len(set(sum(full[:6], []))) == 48


def test_vectorized_histograms_match_adeval():
    rng=np.random.default_rng(10)
    scores=rng.random((100,100),dtype=np.float32)
    scores[0,:3]=[0.,.5,1.]
    label=rng.integers(0,2,size=scores.shape)
    weights=rng.choice([1.,.5,.1,1/7],size=scores.shape).astype(np.float32)
    original=_AccumulateStatCurve(0.,1.,nstrips=1024)
    fast=VectorizedStatCurve(0.,1.,nstrips=1024)
    for accumulator in [original,fast]:
        accumulator.accum(scores,label,weights)
        accumulator.accum(scores,label,1.)
    for name in ['pos_stat','neg_stat','weighted_pos_stat','weighted_neg_stat']:
        np.testing.assert_allclose(getattr(original,name),getattr(fast,name),atol=1e-12,rtol=1e-12)
