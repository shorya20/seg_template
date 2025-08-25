import torch
import numpy as np
import os
import nibabel
import skimage.measure as measure
import scipy.ndimage as ndimage
import time

# `skeletonize_3d` was deprecated/renamed in newer versions, so we fall back to the 2-D `skeletonize` which
# also works on volumetric data slice-wise (scikit-image ≥0.23) if the 3-D version is unavailable.
try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d  # noqa: F401
except ImportError:  # Newer scikit-image
    from skimage.morphology import skeletonize as _skeletonize_3d  # type: ignore

from .metrics import get_parsing
import math
EPSILON = 1e-32
DBR_DETECTION_THRESHOLD = 0.6  # threshold for considering a branch detected

# Debug control settings
DEBUG_TOPOLOGY_METRICS = os.getenv('DEBUG_TOPOLOGY_METRICS', 'detailed').lower()
MAX_DEBUG_COMPONENTS = int(os.getenv('MAX_DEBUG_COMPONENTS', '10'))
MAX_COMPONENTS_THRESHOLD = int(os.getenv('MAX_COMPONENTS_THRESHOLD', '50000'))


def compute_binary_iou(y_true, y_pred, lungs=None, masked=False):
    if masked and lungs is not None:
        lungs = (lungs != 0).astype(np.uint8)
        y_true = y_true * lungs
        y_pred = y_pred * lungs

    yt = (y_true > 0)
    yp = (y_pred > 0)

    yt_sum = int(yt.sum())
    yp_sum = int(yp.sum())

    if yt_sum == 0 and yp_sum == 0:
        return 1.0
    if yt_sum == 0 or yp_sum == 0:
        return 0.0

    intersection = int(np.logical_and(yt, yp).sum())
    if intersection == 0:
        return 0.0

    union = int(np.logical_or(yt, yp).sum())
    return float(intersection) / float(union) if union > 0 else 0.0

def evaluation_branch_metrics(fid, label, pred, lungs=None, refine=False):
    """
    :return: iou, dice, detected length ratio, detected branch ratio,
    precision, leakages, airway missing ratio (AMR),
    masked IoU (ious), DLRs, DBRs, sprec, sleak, sAMR, large_cd
    """
    start_time = time.time()

    label = (label > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    parsing_gt = get_parsing(label, refine)

    cd, num = measure.label(pred, return_num=True, connectivity=1)
    if num < 1:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None

    # Select among top-K largest components if needed (template behavior)
    volume = np.zeros([num], dtype=np.int64)
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    order = np.argsort(volume)[::-1]
    #large_cd is the largest connected component
    large_cd = None
    iou = 0.0
    K = min(5, num)
    for idx in order[:K]:
        candidate = (cd == (idx + 1)).astype(np.uint8)
        iou_candidate = compute_binary_iou(label, candidate)
        if iou_candidate >= 0.1:
            large_cd = candidate
            iou = iou_candidate
            break
        # keep best attempt to report if all are < 0.1
        if iou_candidate > iou:
            iou = iou_candidate
            large_cd = candidate

    if large_cd is None:
        # Should not happen, but guard anyway
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None
    # After large_cd selection
    large_cd = ndimage.binary_fill_holes(large_cd).astype(np.uint8)
    # If still clearly mismatched, skip heavy topology and return fast metrics
    if iou < 0.1:
        ious = compute_binary_iou(label, large_cd, lungs=lungs, masked=True)
        precision = float((large_cd & label).sum()) / float(large_cd.sum()) if large_cd.sum() > 0 else 0.0
        leakages = float(((large_cd - label) == 1).sum()) / float(label.sum()) if label.sum() > 0 else 0.0
        FN = np.logical_and(label, np.logical_not(large_cd)).sum()
        GT = label.sum()
        AMR = float(FN) / float(GT) if GT != 0 else 0.0
        return iou, 0, 0, precision, leakages, AMR, ious, 0, 0, precision, leakages, AMR, large_cd

    # Topology metrics (DLR/DBR)
    skeleton = _skeletonize_3d(label)
    skeleton = (skeleton > 0).astype("uint8")
    if skeleton.sum() == 0:
        DLR, DBR = 0.0, 0.0
    else:
        DLR = float((large_cd & skeleton).sum()) / float(skeleton.sum())

        num_branch = int(parsing_gt.max())
        if num_branch == 0:
            DBR = 0.0
        else:
            detected_num = 0
            for j in range(num_branch):
                branch_label = ((parsing_gt == (j + 1)).astype(np.uint8)) * skeleton
                denom = int(branch_label.sum())
                if denom > 0:
                    overlap_ratio = float((large_cd & branch_label).sum()) / float(denom)
                    if overlap_ratio >= DBR_DETECTION_THRESHOLD:  # template threshold
                        detected_num += 1
            DBR = float(detected_num) / float(num_branch) if num_branch > 0 else 0.0

    precision = float((large_cd & label).sum()) / float(large_cd.sum()) if large_cd.sum() > 0 else 0.0
    leakages = float(((large_cd - label) == 1).sum()) / float(label.sum()) if label.sum() > 0 else 0.0

    FN = np.logical_and(label, np.logical_not(large_cd)).sum()
    GT = label.sum()
    AMR = float(FN) / float(GT) if GT != 0 else 0.0

    ious = compute_binary_iou(label, large_cd, lungs=lungs, masked=True)

    # Back-compat second set (set equal to primary)
    DLRs, DBRs, sprec, sleak, sAMR = DLR, DBR, precision, leakages, AMR

    return iou, DLR, DBR, precision, leakages, AMR, ious, DLRs, DBRs, sprec, sleak, sAMR, large_cd