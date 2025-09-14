import numpy as np
import os
from scipy import ndimage
import skimage.measure as measure
import nibabel
try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d  # noqa: F401
except ImportError:  # Newer scikit-image
    from skimage.morphology import skeletonize as _skeletonize_3d  # type: ignore
import torch
from monai.metrics import DiceMetric
from typing import Tuple, Dict, Any, Optional, List, Union
import torchmetrics
from monai.utils import MetricReduction

def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    if num == 0:
        return np.zeros_like(label, dtype=np.uint8)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

def skeleton_parsing(skeleton):
    # separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)  # 26-neighborhood
    # Count neighbors including self, then subtract self to get degree on skeleton
    nb = ndimage.convolve(skeleton.astype(np.uint8), neighbor_filter, mode="constant", cval=0) * skeleton
    deg = (nb - 1) * skeleton  # neighbors excluding self on the skeleton

    skeleton_parse = skeleton.copy()
    skeleton_parse[deg >= 3] = 0  # break at junctions (>=3 neighbors)
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse,
    structure=con_filter)
    # remove small branches
    for i in range(num):
        a = cd[cd == (i + 1)]
        if a.shape[0] < 3:  # drop stubs smaller than 3 voxels
            skeleton_parse[cd == (i + 1)] = 0
    cd, num = ndimage.label(skeleton_parse,
    structure=con_filter)
    return skeleton_parse, cd, num

def tree_parsing_func(skeleton_parse, label, cd):
    # parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_parsing = np.zeros(label.shape, dtype=np.uint16)
    tree_parsing = cd[inds[0, ...], inds[1, ...], inds[2, ...]] * label
    return tree_parsing

def loc_trachea(tree_parsing, num):
    # find the trachea
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((tree_parsing == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    # print(volume_sort)
    trachea = (volume_sort[-1] + 1)
    return trachea

def get_parsing(mask, refine=False):
    mask = (mask > 0).astype(np.uint8)
    mask = large_connected_domain(mask)
    skeleton = _skeletonize_3d(mask)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    tree_parsing = tree_parsing_func(skeleton_parse, mask, cd)
    return tree_parsing

class AirwayMetrics(torchmetrics.Metric):
    """
    Compute airway-specific metrics for segmentation evaluation.
    This class combines standard segmentation metrics (Dice, IoU) with
    specialized airway topology metrics.
    """
    def __init__(self, include_background=False):
        super().__init__()
        self.include_background = include_background
        self.dice_metric = DiceMetric(
            include_background=include_background, 
            reduction=MetricReduction.MEAN_BATCH
        )
        
        # Register buffers for storing results
        self.add_state("dice_scores", default=[], dist_reduce_fx="cat")
        self.add_state("iou_scores", default=[], dist_reduce_fx="cat")
        self.add_state("trachea_dice_scores", default=[], dist_reduce_fx="cat")
        self.add_state("dlr_scores", default=[], dist_reduce_fx="cat")  # Detected Length Ratio
        self.add_state("dbr_scores", default=[], dist_reduce_fx="cat")  # Detected Branch Ratio
        self.add_state("precision_scores", default=[], dist_reduce_fx="cat")
        self.add_state("leakage_scores", default=[], dist_reduce_fx="cat")
        self.add_state("amr_scores", default=[], dist_reduce_fx="cat")  # Airway Missing Ratio
        
    def update(self, pred: torch.Tensor, target: torch.Tensor, lungs: Optional[torch.Tensor] = None, filenames: Optional[list] = None) -> None:
        """
        Update metric states with predictions and targets.
        
        Args:
            pred: Predicted binary segmentation (B, 1, D, H, W)
            target: Ground truth binary segmentation (B, 1, D, H, W)
            lungs: Optional lung mask for masked evaluation (B, 1, D, H, W)
        """
        if not pred.is_cuda:
            pred = pred.device()
        if not target.is_cuda:
            target = target.device()
        if lungs is not None and not lungs.is_cuda:
            lungs = lungs.device()
        
        # Standard Dice metric (using MONAI's implementation)
        dice_value = self.dice_metric(pred, target)
        self.dice_scores.append(dice_value)
        
        # Process each sample in the batch
        batch_size = pred.shape[0]
        for b in range(batch_size):
            # Get CPU numpy arrays for topology metrics
            p = pred[b, 0].detach().cpu().numpy() > 0.5
            t = target[b, 0].detach().cpu().numpy() > 0.5
            l = None if lungs is None else lungs[b, 0].detach().cpu().numpy()
            
            # Compute binary IoU
            from .topo_metrics import compute_binary_iou
            if l is not None:
                iou = compute_binary_iou(t, p, lungs=l)
                self.iou_scores.append(torch.tensor(float(iou), dtype=torch.float32).cuda())

            try:
                parsing_t = get_parsing(t, refine=False)
                num_branches_t = int(parsing_t.max())
                trachea_idx_t = loc_trachea(parsing_t, num_branches_t)
                trachea_mask_t = (parsing_t == trachea_idx_t).astype(np.uint8)

                parsing_p = get_parsing(p, refine=False)
                num_branches_p = int(parsing_p.max())
                trachea_idx_p = loc_trachea(parsing_p, num_branches_p)
                trachea_mask_p = (parsing_p == trachea_idx_p).astype(np.uint8)

                inter = float(np.logical_and(trachea_mask_t > 0, trachea_mask_p > 0).sum())
                denom = float((trachea_mask_t > 0).sum() + (trachea_mask_p > 0).sum())
                trachea_dice = (2.0 * inter / denom) if denom > 0 else 0.0

                self.trachea_dice_scores.append(torch.tensor(trachea_dice, dtype=torch.float32).cuda())
            except Exception:
                self.trachea_dice_scores.append(torch.tensor(0.0, dtype=torch.float32).cuda())
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics from stored state."""
        # Compute means for all metrics
        if len(self.dice_scores) > 0:
            mean_dice = torch.cat(self.dice_scores).mean()
        else:
            mean_dice = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.iou_scores) > 0:
            mean_iou = torch.stack(self.iou_scores).mean()
        else:
            mean_iou = torch.tensor(0.0, dtype=torch.float32)

        if len(self.trachea_dice_scores) > 0:
            mean_trachea_dice = torch.stack(self.trachea_dice_scores).mean()
        else:
            mean_trachea_dice = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.dlr_scores) > 0:
            mean_dlr = torch.stack(self.dlr_scores).mean()
        else:
            mean_dlr = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.dbr_scores) > 0:
            mean_dbr = torch.stack(self.dbr_scores).mean()
        else:
            mean_dbr = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.precision_scores) > 0:
            mean_precision = torch.stack(self.precision_scores).mean()
        else:
            mean_precision = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.leakage_scores) > 0:
            mean_leakage = torch.stack(self.leakage_scores).mean()
        else:
            mean_leakage = torch.tensor(0.0, dtype=torch.float32)
            
        if len(self.amr_scores) > 0:
            mean_amr = torch.stack(self.amr_scores).mean()
        else:
            mean_amr = torch.tensor(0.0, dtype=torch.float32)
        
        return {
            "dice": mean_dice,
            "iou": mean_iou,
            "trachea_dice": mean_trachea_dice,
            "dlr": mean_dlr,
            "dbr": mean_dbr,
            "precision": mean_precision,
            "leakage": mean_leakage,
            "amr": mean_amr
        }
        
    def reset(self) -> None:
        """Reset all states."""
        self.dice_scores = []
        self.iou_scores = []
        self.trachea_dice_scores = []
        self.dlr_scores = []
        self.dbr_scores = []
        self.precision_scores = []
        self.leakage_scores = []
        self.amr_scores = []