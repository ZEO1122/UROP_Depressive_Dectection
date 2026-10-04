"""Fixed-threshold metrics and participant-paired bootstrap."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def metrics(truth, probability):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid predicted probabilities')
    truth = np.asarray(truth, dtype=int)
    predicted = probability > .5  # argmax([p0,p1]) resolves exact ties to class 0.
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    result = {'n': len(truth), 'macro_f1': float(f1_score(truth, predicted, labels=[0, 1], average='macro', zero_division=0)),
              'positive_f1': float(f1_score(truth, predicted, zero_division=0)),
              'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
              'specificity': float(tn / (tn + fp)) if tn + fp else None,
              'auroc': float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
              'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
    for average in ('macro', 'weighted'):
        result[average + '_precision'] = float(precision_score(truth, predicted, labels=[0, 1], average=average, zero_division=0))
        result[average + '_recall'] = float(recall_score(truth, predicted, labels=[0, 1], average=average, zero_division=0))
    result['weighted_f1'] = float(f1_score(truth, predicted, labels=[0, 1], average='weighted', zero_division=0))
    result['gmean'] = (float(np.sqrt(result['sensitivity'] * result['specificity']))
                       if result['sensitivity'] is not None and result['specificity'] is not None else None)
    return result


def paired_bootstrap(truth, a, b, repeats=2000):
    """Same participant resample for every paired seed, then average seed F1 deltas."""
    truth = np.asarray(truth, dtype=int)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(truth):
        raise ValueError('Expected matched seed-by-participant probability arrays')
    def score(y, p):
        cm = np.bincount(y * 2 + (p > .5).astype(int), minlength=4).reshape(2, 2)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(2 * cm.diagonal(), denom, out=np.zeros(2), where=denom > 0).mean()
    def difference(idx):
        return float(np.mean([score(truth[idx], x[idx]) - score(truth[idx], y[idx]) for x, y in zip(a, b)]))
    rng = np.random.default_rng(91273)
    deltas = [difference(rng.integers(0, len(truth), len(truth))) for _ in range(repeats)]
    return {'delta_macro_f1': difference(np.arange(len(truth))), 'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(),
            'repeats': repeats, 'unit': 'participant, same resample across fixed paired seeds',
            'prediction': 'mean of per-seed Macro-F1 differences; not ensemble probabilities'}

