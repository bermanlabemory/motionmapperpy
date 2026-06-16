"""
General, animal-agnostic pose-preprocessing utilities.

These build the egocentric (translation- and rotation-invariant) postural representations that
feed the wavelet transform. Unlike ``demoutils.egoh5`` (which is hard-coded for the 16-keypoint
mouse demo), these work for any 2-D or 3-D keypoint layout.
"""
import numpy as np


def egocenter(keypoints, center_idx, head_idx, tail_idx=None, flatten=False):
    """Egocenter a pose time series so it depends only on the animal's *shape*.

    Each frame is translated so ``center_idx`` sits at the origin, then rotated within the
    xy-plane so the body axis points along +x. The axis is ``(head_idx - center_idx)``, or
    ``(head_idx - tail_idx)`` when ``tail_idx`` is given. The z coordinate (if present) is kept.

    :param keypoints: ``(T, K, D)`` array of tracked keypoints (``D`` = 2 or 3).
    :param center_idx: keypoint index to place at the origin (e.g. tail-base, body center).
    :param head_idx: keypoint index defining the front of the body axis (e.g. snout).
    :param tail_idx: optional keypoint index for the back of the axis; if ``None`` the axis is
        taken from the center to the head.
    :param flatten: if True, return ``(T, K*D)`` instead of ``(T, K, D)``.
    :return: egocentric keypoints, same shape as input (or flattened).
    """
    P = np.asarray(keypoints, dtype=float)
    if P.ndim != 3:
        raise ValueError('keypoints must have shape (T, K, D); got %s' % (P.shape,))
    Q = P - P[:, [center_idx], :]
    axis = Q[:, head_idx, :2] - (Q[:, tail_idx, :2] if tail_idx is not None else 0.0)
    th = np.arctan2(axis[:, 1], axis[:, 0])
    c, s = np.cos(-th), np.sin(-th)
    x, y = Q[:, :, 0].copy(), Q[:, :, 1].copy()
    Q[:, :, 0] = x * c[:, None] - y * s[:, None]
    Q[:, :, 1] = x * s[:, None] + y * c[:, None]
    return Q.reshape(len(P), -1) if flatten else Q


def pairwiseDistances(keypoints, indices=None):
    """Per-frame pairwise Euclidean distances between keypoints.

    A simple egocentric, rotation-invariant pose representation (the one used in the rodent
    demos): no alignment needed because distances are invariant to translation and rotation.

    :param keypoints: ``(T, K, D)`` array.
    :param indices: optional subset of keypoint indices to use (default: all).
    :return: ``(T, n_pairs)`` array of distances.
    """
    P = np.asarray(keypoints, dtype=float)
    idx = np.arange(P.shape[1]) if indices is None else np.asarray(indices)
    pairs = [(idx[m], idx[n]) for m in range(len(idx)) for n in range(m + 1, len(idx))]
    return np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in pairs], axis=1)
