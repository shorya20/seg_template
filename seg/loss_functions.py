import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from monai.losses import DiceLoss, FocalLoss, DiceCELoss
from monai.losses import SoftclDiceLoss as MONAI_SoftclDiceLoss
from monai.losses import SoftDiceclDiceLoss as MONAI_SoftDiceclDiceLoss
from monai.networks.utils import one_hot
from monai.utils import LossReduction
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List, Optional, Tuple, Union

#helper function as MONAI's SoftclDiceLoss and SoftDiceclDiceLoss expects two channels (background and foreground) input as probabilities.
def _to_two_channel_probs(logits: torch.Tensor, target: torch.Tensor):
    """
    Convert single-channel logits & binary target to 2-channel probabilities and 2-channel targets.
    logits: (B,1,Z,Y,X), target: (B,1,Z,Y,X) in {0,1}
    Returns y_true_2c, y_pred_2c in float32 for stability.
    """
    p = torch.sigmoid(logits)                       # (B,1,.,.,.)
    if target.shape[1] == 1:
        tgt_fg = target
    else:  # if one-hot came in, assume last channel is fg
        tgt_fg = target[:, -1:, ...]
    y_pred_2c = torch.cat([1.0 - p, p], dim=1).float()
    y_true_2c = torch.cat([1.0 - tgt_fg, tgt_fg], dim=1).float()
    return y_true_2c, y_pred_2c


#helper function to calculate the soft skeletonization
def soft_skeletonize(x, thresh_width=10):
    """
    Differentiable skeletonization of a binary image.
    """
    return torch.sigmoid(thresh_width * (x - 0.5)) * (1 - torch.sigmoid(thresh_width * (x - 0.5))) * x

def soft_dilate(x, kernel_size=3):
    """
    Differentiable dilation of a binary image.
    """
    #2D data
    if len(x.shape)==4:
        pool = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    #3D data
    else:
        pool = F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return pool(x)

def compute_dtm(img_gt, pred):
    """
    Compute the distance transform map (DTM) for the ground truth and predicted masks.
    """
    if isinstance(img_gt, torch.Tensor):
        img_gt = img_gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    fg_dtm = np.zeros_like(img_gt) #foreground distance transform map
    for b in range(img_gt.shape[0]):
        for c in range(img_gt.shape[1]):
            posmask = img_gt[b, c].astype(np.bool_)
            if posmask.any():
                fg_dtm[b, c] = distance_transform_edt(posmask)
    return torch.from_numpy(fg_dtm).float().to(img_gt.device)  

def compute_weight_maps(gt, sigma=20, r=5):
    """
    Compute the weight maps for boundary enhancement
    """
    if not isinstance(gt, torch.Tensor):
        gt = torch.from_numpy(gt)
    
    device = gt.device
    weights = torch.zeros_like(gt, dtype=torch.float32)
    gt_np = gt.detach().cpu().numpy()
    for b in range(gt.shape[0]):
        for c in range(gt.shape[1]):#Skip the background channel
            #Distance maps
            dist = distance_transform_edt(gt_np[b, c].cpu().numpy())
            dist = torch.from_numpy(dist).to(device)

            #Weights formula: w = 1 + r * exp(-dist^2 / 2*sigma^2)
            weights[b, c] = 1 + r * torch.exp(-(dist**2) / (2 * sigma ** 2))
    return weights

class HardclDiceLoss(nn.Module):
    """
    HardclDiceLoss: A class that implements the hardclDice loss function.
    This loss function is a combination of the Dice loss and the hardclDice loss.
    """
    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            softmax: bool = False,
            sigmoid: bool = True,
            other_act: Optional[callable] = None,
            squared_pred : bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch: bool = False,
            iter_: int = 3,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            alpha: float = 0.5,
    ) -> None:
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            sigmoid=sigmoid,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.iter_ = iter_
        self.smooth = smooth_nr
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Args: input: predictions
                target: ground truth
        """
        dice = super().forward(input, target)  
        #Centerline Dice calculation
        if self.sigmoid:
            input = torch.sigmoid(input)
            
        #Soft skeletonization
        target_skeleton = soft_skeletonize(target, 10)
        pred_skeleton = soft_skeletonize(input, 10)
        #Iterate to improve skeletonization
        for _ in range(self.iter_):
            target_skeleton = soft_skeletonize(target_skeleton, 10)
            pred_skeleton = soft_skeletonize(pred_skeleton, 10)
        #Calculate intersection and union for centerline Dice
        cl_dice = 0.0
        for b in range(target.shape[0]):
            intersect = torch.sum(target_skeleton[b] * pred_skeleton[b])
            denominator = torch.sum(target_skeleton[b]) + torch.sum(pred_skeleton[b])
            cl_dice_score = (2.0 * intersect + self.smooth) / (denominator + self.smooth)
            cl_dice += 1.0 - cl_dice_score
        
        cl_dice /= target.shape[0]
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice

class SoftclDiceAdapter(nn.Module):
    """
    Adapter around MONAI SoftclDiceLoss for binary (1-channel logit) outputs.
    """
    def __init__(self, iter_: int = 3, smooth: float = 1.0):
        super().__init__()
        self.loss = MONAI_SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_true_2c, y_pred_2c = _to_two_channel_probs(logits, target)
        # MONAI's SoftclDiceLoss expects forward(y_true, y_pred)
        return self.loss(y_true_2c, y_pred_2c)

class SoftDiceclDiceAdapter(nn.Module):
    """
    Adapter around MONAI SoftDiceclDiceLoss (soft dice + clDice inside MONAI).
    """
    def __init__(self, iter_: int = 3, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.loss = MONAI_SoftDiceclDiceLoss(iter_=iter_, alpha=alpha, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_true_2c, y_pred_2c = _to_two_channel_probs(logits, target)
        return self.loss(y_true_2c, y_pred_2c)

class DiceCEPlusSoftclDice(nn.Module):
    """
    Composite: DiceCE(logits, target) + cl_weight * SoftclDice(probs_2c, target_2c).
    Use this as your main 'DiceCE + softCLDice' improvement.
    """
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 0.5,
        cl_weight: float = 0.3,
        cl_iter: int = 3,
        cl_smooth: float = 1.0,
    ):
        super().__init__()
        self.dicece = DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
        self.cldice = MONAI_SoftclDiceLoss(iter_=cl_iter, smooth=cl_smooth)
        self.cl_weight = float(cl_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # DiceCE on logits (BCEWithLogits inside, Dice on sigmoid probs) — safe & standard
        loss_dicece = self.dicece(logits, target)
        # clDice on 2-channel probabilities
        y_true_2c, y_pred_2c = _to_two_channel_probs(logits, target)
        loss_cldice = self.cldice(y_true_2c, y_pred_2c)
        return loss_dicece + self.cl_weight * loss_cldice

class GeneralUnionLoss(nn.Module):
    """
    GeneralUnionLoss: A class that implements the Generalized Union loss function 
    for boundary-aware segmentation and region-based segmentation.
    This loss combines a region-based Dice loss with a boundary-aware term,
    which is weighted using a pre-computed weight_map. This approach is 
    inspired by techniques used in various medical image segmentation challenges,
    such as those seen in top-performing teams that focus on enhancing boundary
    delineation.
    Reference for similar concepts in competitive environments: 
    https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/blob/main/ATM22-Challenge-Top5-Solution/team_timi/train.py
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        alpha: float = 0.2,
        rl: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha  # Weight balancing between region and boundary terms
        self.rl = rl  # Power parameter for boundary enhancement
        self.reduction = reduction
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y, 
            sigmoid=sigmoid,
            softmax=softmax
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: predictions
            target: ground truth
            weight_map: pre-computed weight maps for boundary enhancement
        """
        # Standard Dice loss
        dice_loss = self.dice_loss(input, target)
        
        # Apply sigmoid if specified
        if self.sigmoid:
            input = torch.sigmoid(input)
        
        # Convert target to one-hot if needed
        if self.to_onehot_y:
            target = one_hot(target, num_classes=input.shape[1])
        
        # Boundary term calculation
        boundary_loss = 0.0
        for b in range(input.shape[0]):
            # Calculate weighted loss with boundary enhancement
            weighted_input = input[b] * (weight_map[b] ** self.rl)
            weighted_target = target[b] * (weight_map[b] ** self.rl)
            
            intersect = torch.sum(weighted_input * weighted_target)
            denominator = torch.sum(weighted_input) + torch.sum(weighted_target)
            
            boundary_term = 1.0 - (2.0 * intersect + 1e-5) / (denominator + 1e-5)
            boundary_loss += boundary_term
        
        boundary_loss /= input.shape[0]
        
        # Combine region and boundary terms
        combined_loss = (1.0 - self.alpha) * dice_loss + self.alpha * boundary_loss
        
        return combined_loss

def soft_dice_cldice(
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    iter_: int = 3,
    alpha: float = 0.5,
    smooth: float = 1e-5
):
    """
    Factory function for soft Dice combined with centerline Dice
    """
    dice_loss = DiceLoss(
        include_background=include_background,
        to_onehot_y=to_onehot_y,
        sigmoid=sigmoid,
        softmax=softmax
    )
    
    def loss_function(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Standard Dice Loss
        dice = dice_loss(input, target)
        
        # Apply activation function
        if sigmoid:
            input_sigmoid = torch.sigmoid(input)
        else:
            input_sigmoid = input
        
        # Get soft skeletons
        skel_pred = soft_skeletonize(input_sigmoid)
        skel_gt = soft_skeletonize(target)
        
        # Iterative dilation
        for _ in range(iter_):
            skel_pred = soft_dilate(skel_pred)
            skel_gt = soft_dilate(skel_gt)
        
        # Centerline Dice calculation
        cl_dice = 0.0
        for b in range(target.shape[0]):
            intersect = torch.sum(skel_gt[b] * input_sigmoid[b])
            denominator = torch.sum(skel_gt[b]) + torch.sum(input_sigmoid[b])
            cl_dice_score = (2.0 * intersect + smooth) / (denominator + smooth)
            cl_dice += 1.0 - cl_dice_score
        
        cl_dice /= target.shape[0]
        # Combine standard Dice and centerline Dice
        return (1.0 - alpha) * dice + alpha * cl_dice
    
    return loss_function

class CBEL(nn.Module):
    """
    Context-Based Edge Loss for tubular structure segmentation
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        alpha: List[float] = [0.2, 0.1],  # Weights for region and boundary terms
        rl: List[float] = [0.8, 0.5],     # Power parameters
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha
        self.rl = rl
        self.reduction = reduction
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y, 
            sigmoid=sigmoid,
            softmax=softmax
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor, lung_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: predictions
            target: ground truth
            weight_map: pre-computed weight maps for boundary enhancement
            lung_mask: lung segmentation mask for context awareness
        """
        # Standard Dice loss
        dice_loss = self.dice_loss(input, target)
        
        # Apply sigmoid if specified
        if self.sigmoid:
            input = torch.sigmoid(input)
        
        # Convert target to one-hot if needed
        if self.to_onehot_y:
            target = one_hot(target, num_classes=input.shape[1])
        
        # Edge-based term calculation with lung context
        edge_loss = 0.0
        context_loss = 0.0
        
        for b in range(input.shape[0]):
            # Apply weight maps for boundary enhancement
            weighted_input = input[b] * (weight_map[b] ** self.rl[0])
            weighted_target = target[b] * (weight_map[b] ** self.rl[0])
            
            # Edge term
            edge_intersect = torch.sum(weighted_input * weighted_target)
            edge_denominator = torch.sum(weighted_input) + torch.sum(weighted_target)
            edge_term = 1.0 - (2.0 * edge_intersect + 1e-5) / (edge_denominator + 1e-5)
            edge_loss += edge_term
            
            # Context term (using lung mask to focus on relevant regions)
            context_input = input[b] * lung_mask[b]
            context_target = target[b] * lung_mask[b]
            
            context_intersect = torch.sum(context_input * context_target)
            context_denominator = torch.sum(context_input) + torch.sum(context_target)
            context_term = 1.0 - (2.0 * context_intersect + 1e-5) / (context_denominator + 1e-5)
            context_loss += context_term
        
        edge_loss /= input.shape[0]
        context_loss /= input.shape[0]
        
        # Combine all terms: standard dice, edge-based, and context-based
        combined_loss = (1.0 - self.alpha[0] - self.alpha[1]) * dice_loss + self.alpha[0] * edge_loss + self.alpha[1] * context_loss
        return combined_loss

class AdaptiveTverskyLoss(nn.Module):
    """
    Tversky loss with an adaptive mechanism to switch to Dice-like behavior (alpha=0.5, beta=0.5)
    for patches with very low foreground ratio.
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        alpha: float = 0.3, # Default alpha for Tversky
        beta: float = 0.7,  # Default beta for Tversky
        adaptive_threshold: bool = True, # Enable/disable adaptive behavior
        min_pos_ratio: float = 0.01,  # Threshold for positive ratio to trigger adaptation
        force_static_weights: bool = False, # New: If True, always use init alpha/beta, overriding adaptivity
        smooth: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be between 0 and 1, got {beta}")
        if alpha + beta > 2 or alpha + beta < 0.5: # Basic sanity check, not strictly 1
             print(f"Warning: alpha ({alpha}) + beta ({beta}) is {alpha+beta}. Consider if this is intended.")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.alpha = alpha
        self.beta = beta
        self.adaptive_threshold = adaptive_threshold
        self.min_pos_ratio = min_pos_ratio
        self.force_static_weights = force_static_weights # Store the new param
        self.smooth = smooth
        self.reduction = LossReduction(reduction)

    def _adaptive_adjustment(self, target: torch.Tensor) -> Tuple[float, float]:
        """Adjust alpha and beta based on the positive ratio in the target."""
        # If force_static_weights is True, immediately return the initial alpha and beta
        if self.force_static_weights:
            return self.alpha, self.beta
        
        pos_ratio = torch.mean(target.float())
        
        effective_alpha = self.alpha
        effective_beta = self.beta
        
        # Only apply if adaptive_threshold is also True AND force_static_weights is False (implicitly by reaching here)
        if self.adaptive_threshold and pos_ratio < self.min_pos_ratio:
            print(f"[AdaptiveTverskyLoss Info] Positive ratio ({pos_ratio.item():.6f}) in target is below min_pos_ratio ({self.min_pos_ratio:.6f}). "
                  f"Temporarily switching Tversky alpha/beta from {self.alpha:.2f}/{self.beta:.2f} to 0.5/0.5 for this instance.")
            effective_alpha = 0.5
            effective_beta = 0.5
            
        return effective_alpha, effective_beta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            probs = torch.sigmoid(input)
        elif self.softmax:
            if self.to_onehot_y: # Ensure target is one-hot if softmax output
                 target = one_hot(target, num_classes=input.shape[1])
            probs = F.softmax(input, dim=1)
        else:
            probs = input

        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            if target.shape[1] > 1: # if target is one-hot
                target = target[:, 1:]
        
        # Ensure target is same type as probs
        target = target.type_as(probs)

        # Determine effective alpha and beta
        current_alpha, current_beta = self._adaptive_adjustment(target)

        # Define dimensions to sum over based on input shape
        # Exclude batch and channel dimensions
        dims = tuple(range(2, probs.ndim))

        # True Positives, False Positives, False Negatives
        tp = torch.sum(probs * target, dims)
        fp = torch.sum(probs * (1 - target), dims) # Probs for foreground, target is background
        fn = torch.sum((1 - probs) * target, dims) # Probs for background, target is foreground
        
        # Tversky Index: TI = TP / (TP + α*FP + β*FN)
        tversky_index = (tp + self.smooth) / (tp + current_alpha * fp + current_beta * fn + self.smooth)
        
        # Tversky Loss: 1 - TI
        loss = 1.0 - tversky_index

        # Apply reduction
        if self.reduction == LossReduction.MEAN:
            return loss.mean()
        elif self.reduction == LossReduction.SUM:
            return loss.sum()
        #elif self.reduction == LossReduction.NONE or self.reduction == LossReduction.NONE.value: # MONAI 0.8 vs 0.9+
        #    return loss
        else: # Default to mean if unsure or "none" (which should be handled per item by caller)
            return loss.mean()