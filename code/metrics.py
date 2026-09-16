"""Conventional category-pooled anomaly metrics using ADEval histograms."""
import numpy as np
from adeval import EvalAccumulator
from adeval.mem_effic import _AccumulateStatCurve
from sklearn.metrics import roc_auc_score


class VectorizedStatCurve(_AccumulateStatCurve):
    """Equivalent weighted histograms without looping over defect areas."""

    def accum(self, preds, label, weight=1.):
        preds=np.asarray(preds).reshape(-1)
        label=np.asarray(label).reshape(-1)!=0
        indices=np.searchsorted(self.bins,preds,side='right')-1
        if np.any(indices<0) or np.any(indices>=len(self.pos_stat)):
            raise ValueError('Nonfinite or out-of-range histogram index')
        positive,negative=indices[label],indices[~label]
        pos=np.bincount(positive,minlength=len(self.pos_stat))
        neg=np.bincount(negative,minlength=len(self.neg_stat))
        self.pos_stat+=pos
        self.neg_stat+=neg
        if np.ndim(weight)==0:
            self.weighted_pos_stat+=pos*float(weight)
            self.weighted_neg_stat+=neg*float(weight)
        else:
            weights=np.asarray(weight,dtype=np.float64).reshape(-1)
            self.weighted_pos_stat+=np.bincount(positive,weights=weights[label],minlength=len(pos))
            self.weighted_neg_stat+=np.bincount(negative,weights=weights[~label],minlength=len(neg))


def partial_area(fpr, overlap, limit):
    """Integrate a monotone threshold curve, retaining vertical segments."""
    if not 0 < limit <= 1:
        raise ValueError('The integration limit must be in (0, 1]')
    fpr = np.asarray(fpr, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    assert np.all(np.diff(fpr) >= 0)
    index = np.searchsorted(fpr, limit, side='right')
    x, y = fpr[:index], overlap[:index]
    if x[-1] < limit:
        endpoint = np.interp(limit, fpr[index-1:index+1], overlap[index-1:index+1])
        x, y = np.r_[x, limit], np.r_[y, endpoint]
    return float(np.trapezoid(y, x) / limit)


class PixelMetrics:
    def __init__(self, lower, upper, bins=262144):
        margin = max(abs(lower), abs(upper), 1.) * 1e-5
        self.acc = EvalAccumulator(float(lower-margin), float(upper+margin), nstrips=bins)
        self.acc._pixel_acc = VectorizedStatCurve(float(lower-margin),float(upper+margin),nstrips=bins)
        self.bins = bins

    def add(self, prediction, target):
        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.uint8)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite anomaly map')
        self.acc.add_anomap(prediction, target)

    def summary(self, pro_limit):
        pixel = self.acc._pixel_acc
        positives = pixel.pos_stat.sum()
        negatives = pixel.neg_stat.sum()
        regions = pixel.weighted_pos_stat.sum()
        if positives <= 0 or negatives <= 0 or regions <= 0:
            raise ValueError('Pixel metrics require positive and background pixels')
        tp = np.r_[0., np.cumsum(pixel.pos_stat[::-1])]
        fp = np.r_[0., np.cumsum(pixel.neg_stat[::-1])]
        weighted_tp = np.r_[0., np.cumsum(pixel.weighted_pos_stat[::-1])]
        recall, fpr = tp / positives, fp / negatives
        precision = np.divide(tp, tp+fp, out=np.ones_like(tp), where=tp+fp > 0)
        # Stepwise recall integration is AP, unlike trapezoidal PR AUC.
        ap = np.sum(np.diff(recall) * precision[1:])
        pro = weighted_tp / regions
        return {'p_auroc': float(np.trapezoid(recall, fpr)), 'p_ap': float(ap),
                'aupro': partial_area(fpr, pro, pro_limit), 'aupro_limit': pro_limit,
                'histogram_bins': self.bins, 'positive_pixels': int(positives),
                'background_pixels': int(negatives), 'defect_regions': int(round(regions))}


def image_auroc(labels, scores):
    if len(set(labels)) != 2:
        raise ValueError('Image AUROC requires normal and anomalous examples')
    return float(roc_auc_score(labels, scores))
