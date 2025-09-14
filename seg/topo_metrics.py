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

from .metrics import get_parsing, skeleton_parsing, tree_parsing_func, large_connected_domain
import math
EPSILON = 1e-32
DBR_DETECTION_THRESHOLD = float(os.getenv("DBR_DETECTION_THRESHOLD", "0.8")) #Reference: https://arxiv.org/pdf/2507.06581
# EXACT'09-style minimum recovered centerline length to count a branch as detected. Reference: EXACT'09 paper Lo, P. and van Ginneken, B. and Reinhardt, J M. and Tarunashree, Y. and Jong, Irving B.;  Extraction of Airways From CT (EXACT’09); IEEE Transactions on Medical Imaging, 2012.
DBR_MIN_HIT_MM = float(os.getenv("DBR_MIN_HIT_MM", "1.0"))

# Debug control settings (read at import time)
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


def evaluation_branch_metrics(fid, label, pred, lungs=None, refine=False, spacing_zyx_mm=None):
    """
    :return: iou, dice, detected length ratio, detected branch ratio,
    precision, leakages, airway missing ratio (AMR),
    masked IoU (ious), DLRs, DBRs, sprec, sleak, sAMR, large_cd
    """
    start_time = time.time()

    label = (label > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    # Note: No early cropping or fast-path returns here; keep original behavior

    # Build branch parsing once without re-skeletonizing multiple times
    # Follow same steps as get_parsing but reuse pieces locally to avoid duplicate work elsewhere
    # 1) Restrict to largest connected component (airway tree)
    mask_lcc = large_connected_domain((label > 0).astype(np.uint8))
    # 2) Skeletonize LCC only (consistent with get_parsing semantics)
    skel = _skeletonize_3d(mask_lcc)
    skel = (skel > 0).astype(np.uint8)
    # 3) Parse branches on skeleton and propagate labels into mask
    skel_parse, skel_cd, _skel_num = skeleton_parsing(skel)
    parsing_gt = tree_parsing_func(skel_parse, mask_lcc, skel_cd)

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
            sel_mark = "SELECTED" if lab == selected_lab else ""
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
    #step size in mm for centerline voxels (fallback 1.0 mm/vox)
    step_mm = float(np.mean(np.asarray(spacing_zyx_mm))) if spacing_zyx_mm is not None else 1.0
    min_hit_vox = max(1, int(round(DBR_MIN_HIT_MM / max(step_mm, 1e-8))))
    # Topology metrics (DLR/DBR) on LCC skeleton to match parsing
    skeleton = skel  # already uint8
    if skeleton.sum() == 0:
        DLR, DBR = 0.0, 0.0
    else:
        DLR = float((large_cd & skeleton).sum()) / float(max(int(skeleton.sum()), 1))

        num_branch = int(parsing_gt.max())
        # Vectorized DBR computation
        if num_branch == 0:
            DBR = 0.0
            detected_num = 0
        else:
            # Labels for each skeleton voxel (0 for background)
            branch_ids = (parsing_gt.astype(np.int32)) * (skeleton.astype(np.int32))
            # Denominator per branch: number of skeleton voxels per branch
            denom_counts = np.bincount(branch_ids.ravel(), minlength=num_branch + 1)
            denom_per_branch = denom_counts[1:]

            # One-time dilation tolerance (~1 mm) outside the loop
            pred_centerline = (large_cd > 0)
            tol = max(1, int(round(1.0 / step_mm)))  # ~1 mm tolerance
            pred_centerline_dil = ndimage.binary_dilation(pred_centerline, iterations=tol)

            # Hits: skeleton voxels whose branch id > 0 and overlap with dilated prediction
            overlap_mask = (branch_ids > 0) & pred_centerline_dil
            hit_labels = branch_ids[overlap_mask]
            hit_counts = np.bincount(hit_labels.ravel(), minlength=num_branch + 1)[1:]

            # Avoid division by zero; branches with denom==0 are ignored in detection
            with np.errstate(divide='ignore', invalid='ignore'):
                overlap_ratio = np.zeros_like(hit_counts, dtype=np.float64)
                valid = denom_per_branch > 0
                overlap_ratio[valid] = hit_counts[valid] / denom_per_branch[valid]

            detected_num = int(np.sum(overlap_ratio >= DBR_DETECTION_THRESHOLD))
            DBR = float(detected_num) / float(num_branch)

        # Optional lightweight debug print without per-branch loops
        if DEBUG_TOPOLOGY_METRICS in ('minimal', 'detailed', 'full'):
            # Compute median branch length via bincount on skeleton once
            branch_ids_dbg = (parsing_gt.astype(np.int32)) * (skeleton.astype(np.int32))
            lengths_dbg = np.bincount(branch_ids_dbg.ravel())
            median_branch_len = float(np.median(lengths_dbg[1:])) if lengths_dbg.size > 1 else 0.0
            print(
                f"DEBUG eval_metrics: num_branch={num_branch}, median_branch_len={median_branch_len:.1f}"
            )
            print(
                f"DEBUG eval_metrics: DLR={DLR:.4f}, DBR={DBR:.4f} "
                f"(detected {detected_num}/{num_branch} branches with min_hit_vox={min_hit_vox}, step_mm={step_mm:.3f})"
            )
    precision = float((large_cd & label).sum()) / float(large_cd.sum()) if large_cd.sum() > 0 else 0.0
    leakages = float(((large_cd - label) == 1).sum()) / float(label.sum()) if label.sum() > 0 else 0.0

    FN = np.logical_and(label, np.logical_not(large_cd)).sum()
    GT = label.sum()
    AMR = float(FN) / float(GT) if GT != 0 else 0.0

    ious = compute_binary_iou(label, large_cd, lungs=lungs, masked=True)

    # Back-compat second set (set equal to primary)
    DLRs, DBRs, sprec, sleak, sAMR = DLR, DBR, precision, leakages, AMR

    return iou, DLR, DBR, precision, leakages, AMR, ious, DLRs, DBRs, sprec, sleak, sAMR, large_cd
