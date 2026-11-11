"""Evaluation arithmetic only; no TEST selection, label resizing or model fitting.

Ranking uses original labels/scores. Coverage-aware threshold recall is a separate
protocol metric: completely ROI-invisible positive sources count as false negatives.
"""
import heapq
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter


def _raw_integer_guard(value):
    """Inspect still-observable raw integers BEFORE NumPy common-type coercion.

    Already-floating ndarrays have no recoverable integer history. Numeric
    ndarray dtypes are checked below; lists/tuples retain their Python scalars.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'iu' and value.size and (int(value.min()) < -(2**53) or int(value.max()) > 2**53):
            raise ValueError('Integer scores outside exact float64 range [-2^53,2^53] are unsupported')
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _raw_integer_guard(item)
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if not -(2**53) <= int(value) <= 2**53:
            raise ValueError('Integer scores outside exact float64 range [-2^53,2^53] are unsupported')


def _scores_labels(scores, labels):
    _raw_integer_guard(scores)
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != labels.shape or not scores.size or scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all():
        raise ValueError('Nonempty matching finite scores and binary labels required')
    # Integer ordering must survive our explicit f8 shard representation.
    if scores.dtype.kind in 'iu' and (int(scores.min()) < -(2**53) or int(scores.max()) > 2**53):
        raise ValueError('Integer scores outside exact float64 range [-2^53,2^53] are unsupported')
    if scores.dtype.kind == 'f' and scores.dtype.itemsize > 8:
        raise ValueError('Floating scores wider than float64 are unsupported')
    if labels.dtype.kind not in 'biu' or not np.isin(labels, [0, 1]).all():
        raise ValueError('Only binary labels required')
    return scores.astype(np.float64).ravel(), labels.astype(bool).ravel()


def grouped_scores(scores, labels):
    """Descending exact-value tie groups: score, positive count, negative count."""
    scores, labels = _scores_labels(scores, labels)
    order = np.argsort(-scores, kind='stable')
    scores, labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1])+1]
    sizes = np.diff(np.r_[starts, len(scores)])
    positive = np.add.reduceat(labels.astype(np.int64), starts)
    result = np.empty(len(starts), dtype=[('score', '<f8'), ('positive', '<i8'), ('negative', '<i8')])
    result['score'], result['positive'], result['negative'] = scores[starts], positive, sizes-positive
    return result


def _ranking_groups(groups):
    groups = list(groups) if not isinstance(groups, np.ndarray) else groups
    positives = sum(int(row[1]) for row in groups)
    negatives = sum(int(row[2]) for row in groups)
    if not positives or not negatives:
        return {'auroc': None, 'average_precision': None, 'positive_count': positives,
                'negative_count': negatives, 'undefined_reason': 'Ranking requires both original classes'}
    tp, fp, roc, ap = 0, 0, 0., 0.
    for _, positive, negative in groups:
        old_tp, old_fp = tp, fp
        tp += int(positive);fp += int(negative)
        roc += (fp-old_fp)/negatives*(tp+old_tp)/(2*positives)
        ap += int(positive)/positives*tp/(tp+fp)
    return {'auroc': float(roc), 'average_precision': float(ap), 'positive_count': positives,
            'negative_count': negatives, 'tie_policy': 'All identical scores enter one threshold group; noninterpolated AP'}


def ranking_metrics(scores, labels):
    return _ranking_groups(grouped_scores(scores, labels))


def write_score_shard(path, scores, labels):
    """Store a per-image/class/level sorted tie-group shard, numeric NPY only."""
    path = Path(path)
    if path.suffix != '.npy':
        raise ValueError('Explicit .npy shard required')
    groups = grouped_scores(scores, labels)
    with path.open('xb') as f:
        np.save(f, groups, allow_pickle=False)
    return {'positive_count': int(groups['positive'].sum()), 'negative_count': int(groups['negative'].sum()),
            'score_groups': len(groups), 'scope': 'observed original-grid common ROI only; outside unobserved'}


def merged_score_groups(paths):
    """Disk-backed merge; combine ties ACROSS shards before ranking arithmetic.

    One NPY mmap per shard; this generator does not concatenate pixel arrays.
    Runtime/open-file limits need a real data resource profile before large jobs.
    """
    arrays, heap = [], []
    dtype = np.dtype([('score', '<f8'), ('positive', '<i8'), ('negative', '<i8')])
    for index, path in enumerate(paths):
        value = np.load(path, allow_pickle=False, mmap_mode='r')
        if value.ndim != 1 or not len(value) or value.dtype != dtype:
            raise ValueError('Exact numeric sorted-group shard format required')
        if not np.isfinite(value['score']).all() or np.any(value['positive'] < 0) or np.any(value['negative'] < 0) or np.any(value['positive']+value['negative'] <= 0):
            raise ValueError('Malformed shard counts/scores')
        if np.any(value['score'][1:] >= value['score'][:-1]):
            raise ValueError('Shard must have strictly descending grouped scores')
        arrays.append(value)
        heapq.heappush(heap, (-float(value[0]['score']), index, 0))
    while heap:
        negative_score = heap[0][0]
        positive, negative = 0, 0
        while heap and heap[0][0] == negative_score:
            _, index, row = heapq.heappop(heap)
            value = arrays[index]
            positive += int(value[row]['positive']);negative += int(value[row]['negative'])
            if row+1 < len(value):
                heapq.heappush(heap, (-float(value[row+1]['score']), index, row+1))
        yield (-negative_score, positive, negative)


def ranking_from_shards(paths):
    # Two bounded passes over mmap groups: totals first, then accumulation.
    paths = tuple(paths)
    positives = negatives = 0
    for _, p, n in merged_score_groups(paths):
        positives += p;negatives += n
    if not positives or not negatives:
        return {'auroc': None, 'average_precision': None, 'positive_count': positives,
                'negative_count': negatives, 'undefined_reason': 'Ranking requires both original classes'}
    tp = fp = 0
    roc = ap = 0.
    for _, p, n in merged_score_groups(paths):
        old_tp = tp;tp += p;fp += n
        roc += n/negatives*(tp+old_tp)/(2*positives)
        ap += p/positives*tp/(tp+fp)
    return {'auroc': float(roc), 'average_precision': float(ap), 'positive_count': positives,
            'negative_count': negatives, 'tie_policy': 'Exact identical-score groups merged across all shards'}


def freeze_normal_threshold(valid_normal_rows, quantile=.95):
    """Empirical higher quantile, NOT a finite-sample conformal guarantee.

    Apply anomaly iff score > threshold; ties at threshold are normal. Call per
    class/level/branch/seed; input must contain distinct normal VALID sources.
    """
    if quantile != .95:
        raise ValueError('This candidate protocol freezes exactly quantile=.95')
    rows = list(valid_normal_rows)
    if not rows or any(row.get('split_name') != 'valid' or row.get('original_image_label') != 'normal' for row in rows):
        raise ValueError('Threshold may use only independent valid normal sources')
    ids = [row.get('source_image_id') for row in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Distinct explicit source IDs required within a threshold group')
    scores, _ = _scores_labels([row['image_score'] for row in rows], np.zeros(len(rows), dtype=bool))
    return {'image_threshold': float(np.quantile(scores, .95, method='higher')), 'quantile': .95,
            'method': 'empirical numpy.quantile higher; no conformal coverage guarantee',
            'anomaly_decision': 'score > image_threshold; ties normal', 'valid_normal_source_ids': ids,
            'valid_normal_count': len(rows)}


def threshold_image_statistics(score, original_label, threshold, *, original_defect_pixels, retained_defect_pixels):
    if original_label not in ('normal', 'anomaly') or not np.isfinite([score, threshold]).all():
        raise ValueError('Original binary image label and finite score/threshold required')
    if any(type(v) is not int or v < 0 for v in (original_defect_pixels, retained_defect_pixels)) or retained_defect_pixels > original_defect_pixels:
        raise ValueError('Valid original/retained defect counts required')
    positive = original_label == 'anomaly'
    if positive != (original_defect_pixels > 0):
        raise ValueError('Original label/mask inconsistency')
    predicted = bool(score > threshold)
    invisible = positive and retained_defect_pixels == 0
    covered_tp = positive and predicted and not invisible
    return {'original_positive': int(positive), 'predicted_positive': int(predicted),
            'tp': int(positive and predicted), 'fp': int(not positive and predicted),
            'fn': int(positive and not predicted), 'tn': int(not positive and not predicted),
            'all_defects_outside_roi': invisible,
            'coverage_aware_tp': int(covered_tp), 'coverage_aware_fn': int(positive and not covered_tp),
            'ranking_score_unchanged': float(score)}


def pixel_statistics(scores_original_roi, mask_original_roi, threshold=None):
    original_mask = np.asarray(mask_original_roi)
    if (original_mask.dtype.kind not in 'biu' or not np.isfinite(original_mask).all()
            or np.any(original_mask < 0) or original_mask.shape != np.asarray(scores_original_roi).shape):
        raise ValueError('Original ROI mask must match scores and contain finite nonnegative integer/bool classes')
    scores, truth = _scores_labels(scores_original_roi, original_mask != 0)
    result = {'observed_pixels': len(truth), 'observed_positive_pixels': int(truth.sum()),
              'observed_negative_pixels': int((~truth).sum()), 'scope': 'original integer pixel centers in common ROI only'}
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError('Finite frozen normal-valid pixel threshold required')
        pred = scores > threshold
        result.update(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()),
                      fn=int((~pred & truth).sum()), tn=int((~pred & ~truth).sum()))
    return result


def restoration_metrics(prediction, target, valid_mask):
    """Unclipped linear RGB, range=1. SSIM Gaussian11/sigma1.5/pop covariance.

    PSNR pools valid RGB scalar errors. SSIM averages RGB scalar local scores
    only where the complete 11x11 window is in the common availability mask.
    Perfect PSNR is represented by null plus perfect=True for strict JSON.
    """
    p, t, mask = np.asarray(prediction), np.asarray(target), np.asarray(valid_mask)
    if p.shape != t.shape or p.ndim != 3 or p.shape[-1] != 3 or mask.shape != p.shape[:2] or mask.dtype != bool or not mask.any() or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Finite matching HWC RGB and nonempty boolean common mask required')
    p, t = p.astype(np.float64), t.astype(np.float64)
    residual = (p-t)[mask]
    sse = float(np.square(residual).sum());elements = residual.size
    mse = sse/elements
    result = {'mse': mse, 'sum_squared_error': sse, 'valid_rgb_elements': elements,
              'psnr_db': float(-10*np.log10(mse)) if mse > 0 else None, 'perfect': mse == 0,
              'data_range': 1., 'domain': 'unclipped linear RGB', 'ssim': None, 'ssim_rgb_windows': 0,
              'ssim_definition': 'Gaussian 11x11 sigma1.5 truncate3.5, K1=.01 K2=.03, population covariance, complete valid windows only'}
    eligible = minimum_filter(mask.astype(np.uint8), size=11, mode='constant', cval=0).astype(bool)
    if not eligible.any():
        result['ssim_undefined_reason'] = 'No complete valid 11x11 window'
        return result
    f = lambda value: gaussian_filter(value, sigma=(1.5,1.5,0), truncate=3.5, mode='reflect')
    up, ut = f(p), f(t)
    vp, vt, cov = f(p*p)-up*up, f(t*t)-ut*ut, f(p*t)-up*ut
    score = ((2*up*ut+.01**2)*(2*cov+.03**2))/((up*up+ut*ut+.01**2)*(vp+vt+.03**2))
    result['ssim'] = float(score[eligible].mean());result['ssim_rgb_windows'] = int(eligible.sum()*3)
    return result


def detection_encoding(linear):
    """Explicit clip only on detection branch, then one standard sRGB transfer."""
    x = np.asarray(linear)
    if x.ndim != 3 or x.shape[-1] != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError('Finite HWC RGB required')
    clipped = np.clip(x.astype(np.float64), 0, 1)
    encoded = np.where(clipped <= .0031308, 12.92*clipped, 1.055*clipped**(1/2.4)-.055)
    affected = (x < 0) | (x > 1)
    return encoded.astype(np.float32), {'clipped_rgb_components': int(affected.sum()),
        'total_rgb_components': x.size, 'clipped_pixels': int(affected.any(axis=-1).sum()),
        'total_pixels': x.shape[0]*x.shape[1], 'policy': 'explicit [0,1] clipping for detection only; restoration metrics unclipped'}
