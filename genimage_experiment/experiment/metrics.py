import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)


def select_threshold(y, p):
    """Maximize validation balanced accuracy; ties prefer threshold nearest 0.5."""
    y, p = np.asarray(y), np.asarray(p)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold selection requires both validation classes.')
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    scores = (1 + tpr[finite] - fpr[finite]) / 2
    thresholds = thresholds[finite]
    candidates = thresholds[np.isclose(scores, scores.max(), atol=1e-12, rtol=0)]
    return float(candidates[np.argmin(abs(candidates - 0.5))])


def metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Evaluation groups must include both classes.')
    if not np.isfinite(p).all():
        raise ValueError('Non-finite prediction probability.')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                real_fpr=float(fp/(tn+fp)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=threshold)
