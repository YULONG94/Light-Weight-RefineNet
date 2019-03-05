import numpy as np


def fast_cm(preds, gt, n_classes):
    """Computing confusion matrix faster.

    Args:
      preds (Tensor) : predictions (either flatten or of size (len(gt), top-N)).
      gt (Tensor) : flatten gt.
      n_classes (int) : number of classes.

    Returns:

      Confusion matrix
      (Tensor of size (n_classes, n_classes)).

    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int_)
    n = gt.shape[0]

    for i in range(n):
        a = gt[i]
        p = preds[i]
        cm[a, p] += 1
    return cm


def compute_iu(cm):
    """Compute IU from confusion matrix.

    Args:
      cm (Tensor) : square confusion matrix.

    Returns:
      IU vector (Tensor).

    """
    pi = 0
    gi = 0
    ii = 0
    denom = 0
    n_classes = cm.shape[0]
    IU = np.ones(n_classes)

    for i in range(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU
