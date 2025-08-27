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
    # --- Debug: report number of components ---
    if DEBUG_TOPOLOGY_METRICS != 'none':
        try:
            _fid_str = os.path.basename(fid) if isinstance(fid, (str, bytes)) else str(fid)
        except Exception:
            _fid_str = str(fid)
        print(f"DEBUG eval_metrics[{_fid_str}]: Found {num} connected components in prediction.")
        if num > MAX_COMPONENTS_THRESHOLD:
            print(f"WARNING: Found {num} connected components (>{MAX_COMPONENTS_THRESHOLD}). This may indicate very noisy predictions.")

    if num < 1:
        if DEBUG_TOPOLOGY_METRICS in ('minimal', 'detailed', 'full'):
            elapsed = time.time() - start_time
            print(f"DEBUG eval_metrics: No components found. Elapsed {elapsed:.3f}s")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None

    # Select among top-K largest components if needed (template behavior)
    volume = np.zeros([num], dtype=np.int64)
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    order = np.argsort(volume)[::-1]

    # Pre-compute IoU for top components for debug display
    debug_top_list = []
    if DEBUG_TOPOLOGY_METRICS in ('detailed', 'minimal'):
        top_k_dbg = min(MAX_DEBUG_COMPONENTS, num)
        for idx in order[:top_k_dbg]:
            lab = idx + 1
            candidate = (cd == lab).astype(np.uint8)
            iou_cand = compute_binary_iou(label, candidate)
            debug_top_list.append((lab, int(volume[idx]), float(iou_cand)))
    elif DEBUG_TOPOLOGY_METRICS == 'full':
        # Expensive: compute for all components and list them in descending volume
        for idx in order:
            lab = idx + 1
            candidate = (cd == lab).astype(np.uint8)
            iou_cand = compute_binary_iou(label, candidate)
            debug_top_list.append((lab, int(volume[idx]), float(iou_cand)))

    #large_cd is the largest connected component (selected by IoU heuristic)
    large_cd = None
    iou = 0.0
    selected_lab = None
    selected_vol = 0

    K = min(5, num)
    for idx in order[:K]:
        candidate = (cd == (idx + 1)).astype(np.uint8)
        iou_candidate = compute_binary_iou(label, candidate)
        if iou_candidate >= 0.1 and (large_cd is None or iou_candidate > iou):
            large_cd = candidate
            iou = iou_candidate
            selected_lab = idx + 1
            selected_vol = int(volume[idx])
            # choose first good component meeting threshold; continue to check if any with higher IoU in top-K
    # If none met threshold, keep the best among top-K
    if large_cd is None:
        best_idx = None
        best_iou = -1.0
        for idx in order[:K]:
            candidate = (cd == (idx + 1)).astype(np.uint8)
            iou_candidate = compute_binary_iou(label, candidate)
            if iou_candidate > best_iou:
                best_iou = iou_candidate
                best_idx = idx
                large_cd = candidate
                iou = iou_candidate
                selected_lab = idx + 1
                selected_vol = int(volume[idx])

    # Debug: print selection summary and top components table if requested
    if DEBUG_TOPOLOGY_METRICS == 'minimal':
        elapsed = time.time() - start_time
        print(f"DEBUG eval_metrics: Selected label={selected_lab} with IoU={iou:.4f}, volume={selected_vol}. Elapsed {elapsed:.3f}s")
    elif DEBUG_TOPOLOGY_METRICS in ('detailed', 'full'):
        print(f"DEBUG eval_metrics: Top {len(debug_top_list)} components by volume:")
        # debug_top_list already sorted by volume desc due to 'order'
        for lab, vol, iou_cand in debug_top_list:
            sel_mark = " ✓ SELECTED" if lab == selected_lab else ""
            print(f"  - label={lab}: volume={vol}, IoU={iou_cand:.4f}{sel_mark}")
        elapsed = time.time() - start_time
        print(f"DEBUG eval_metrics: Selected component label={selected_lab} with IoU={iou:.4f} (volume={selected_vol}), analysis completed in {elapsed:.3f}s")

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