import math, gzip, pickle, os
from pathlib import Path
import os
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
import torch
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, MaskedDiceLoss, DiceFocalLoss, GeneralizedDiceLoss
from .loss_functions import HardclDiceLoss, GeneralUnionLoss, soft_dice_cldice, CBEL , SoftclDiceAdapter, SoftDiceclDiceAdapter, DiceCEPlusSoftclDice
try:
    from models.WingsNet import WingsNet
    from models.ResUNet import ResUNet
except ImportError:
    import sys
    import os.path as path
    parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from models.WingsNet import WingsNet
    from models.ResUNet import ResUNet
from monai.metrics import DiceMetric, MeanIoU
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import BasicUNet, UNet, AttentionUnet
from monai.utils import set_determinism, MetricReduction, Method
import random
import pickle
import sys
import nibabel as nib
from monai.transforms import SpatialResample, get_largest_connected_component_mask
from scipy.ndimage import label as _ndlabel
from scipy.ndimage import binary_fill_holes as _fill
from scipy.ndimage import distance_transform_edt
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

#Set seed 
random.seed(1221)
np.random.seed(1221)
# safer determinism setting to avoid CUDA errors
try:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(seed=0)
    print("Utils: Using deterministic algorithms with fallback for CUDA compatibility")
except Exception as e:
    # Fallback: set seeds but allow non-deterministic algorithms
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    print(f"Utils: Warning - Using non-deterministic algorithms for CUDA compatibility: {e}")

def keep_topk_components(mask_zyx: np.ndarray, k: int = 2, fill_holes: bool = True) -> np.ndarray:
    m = (mask_zyx > 0).astype(np.uint8)
    lab, n = _ndlabel(m)
    if n <= k: 
        return _fill(m).astype(np.uint8) if fill_holes else m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.argsort(sizes)[-k:]
    out = np.isin(lab, keep)
    return _fill(out).astype(np.uint8) if fill_holes else out.astype(np.uint8)

def estimate_memory_training(model, sample_input, optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False,
                             device=0):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    optimizer = optimizer_type(model.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    # model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = model(sample_input.to(device)).sum()
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b) * amp_multiplier
    gradient_memory = model_memory
    if isinstance(optimizer, torch.optim.Adam):
        o = 2
    elif isinstance(optimizer, torch.optim.RMSprop):
        o = 1
    elif isinstance(optimizer, torch.optim.SGD):
        o = 0
    elif isinstance(optimizer, torch.optim.Adagrad):
        o = 1
    else:
        raise ValueError("Unsupported optimizer. Look up how many moments are" +
                         "stored by your optimizer and add a case to the optimizer checker.")
    gradient_moment_memory = o * gradient_memory
    total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory

    return total_memory


def estimate_memory_inference(model, sample_input, batch_size=1, use_amp=False, device=0):
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = model_input.to(device)
    total_memory = model_memory

    return total_memory


def test_memory_training(model=None, sample_input=None, in_size=100, out_size=10, hidden_size=100,
                         optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False, device=0):
    # sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    max_mem_est = estimate_memory_training(model, sample_input, optimizer_type=optimizer_type, batch_size=batch_size,
                                           use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    optimizer = optimizer_type(model.parameters(), lr=.001)
    print("Beginning mem:", torch.cuda.memory_allocated(device),
          "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    for i in range(3):
        optimizer.zero_grad()
        print("Iteration", i)
        with torch.cuda.amp.autocast(enabled=use_amp):
            a = torch.cuda.memory_allocated(device)
            out = model(sample_input.to(device)).sum()  # Taking the sum here just to get a scalar output
            b = torch.cuda.memory_allocated(device)
        print("1 - After forward pass", torch.cuda.memory_allocated(device))
        print("2 - Memory consumed by forward pass", b - a)
        out.backward()
        print("3 - After backward pass", torch.cuda.memory_allocated(device))
        optimizer.step()
        print("4 - After optimizer step", torch.cuda.memory_allocated(device))


def test_memory_inference(in_size=100, out_size=10, hidden_size=100, batch_size=1, use_amp=False, device=0):
    sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(in_size, hidden_size),
                          *[nn.Linear(hidden_size, hidden_size) for _ in range(200)],
                          nn.Linear(hidden_size, out_size))
    max_mem_est = estimate_memory_inference(model, sample_input[0], batch_size=batch_size, use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    print("Beginning mem:", torch.cuda.memory_allocated(device),
          "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    with torch.no_grad():
        for i in range(3):
            print("Iteration", i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                a = torch.cuda.memory_allocated(device)
                out = model(sample_input.to(device)).sum()  # Taking the sum here just to get a scalar output
                b = torch.cuda.memory_allocated(device)
            print("1 - After forward pass", torch.cuda.memory_allocated(device))
            print("2 - Memory consumed by forward pass", b - a)


def save_to_check(outputs_path, filename, epoch, inputs, labels, lungs, weights):
    to_save = {}
    to_save['inputs'] = inputs
    to_save['labels'] = labels
    to_save['lungs'] = lungs
    to_save['weights'] = weights
    pickle_file_path = outputs_path + f"{filename}_epoch{int(epoch)}.pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(to_save, file)

def get_model(params, device):
    print(f"Creating model: {params['MODEL_NAME']}")
    model = None
    is_training_phase = params.get('TRAINING_PHASE', True)
    
    if params['MODEL_NAME'] == 'ResUNet':
        from monai.networks.layers.factories import Act, Norm
        norm_type = params.get('NORM', 'INSTANCE')
        if isinstance(norm_type, str):
            norm_enum = getattr(Norm, norm_type.upper(), Norm.INSTANCE)
        else:
            norm_enum = norm_type
            
        model = ResUNet(
            spatial_dims=params.get('SPATIAL_DIMS', 3),
            in_channels=params.get('IN_CHANNELS', 1),
            out_channels=params.get('OUT_CHANNELS', 1),
            channels=params.get('CHANNELS', 16),
            strides=params.get('STRIDES', 2),
            num_res_units=params.get('NUM_RES_UNITS', 1),  
            norm=norm_enum,
            act=Act.RELU,  # Default to ReLU
            dropout=params.get('DROPOUT', 0.1),
            upsample_mode='convtranspose'  # Default upsampling mode
        )
        print(f"Created ResUNet with channels={params.get('CHANNELS', 16)}, dropout={params.get('DROPOUT', 0.1)}")
        
    elif params['MODEL_NAME'] == 'SimpleResUNet':
        # Use the simplified ResUNet implementation without attention gates
        try:
            from models.SimpleResUNet import SimpleResUNet
        except ImportError:
            # Fallback with improved path handling
            import sys
            import os.path as path
            parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from models.SimpleResUNet import SimpleResUNet
        from monai.networks.layers.factories import Act, Norm
        
        # Convert string norm to enum if needed
        norm_type = params.get('NORM', 'INSTANCE')
        if isinstance(norm_type, str):
            norm_enum = getattr(Norm, norm_type.upper(), Norm.INSTANCE)
        else:
            norm_enum = norm_type
            
        model = SimpleResUNet(
            spatial_dims=params.get('SPATIAL_DIMS', 3),
            in_channels=params.get('IN_CHANNELS', 1),
            out_channels=params.get('OUT_CHANNELS', 1),
            channels=params.get('CHANNELS', 32),
            strides=params.get('STRIDES', 2),
            norm=norm_enum,
            act=Act.RELU,
            dropout=params.get('DROPOUT', 0.1)
        )
        print(f"Created SimpleResUNet with channels={params.get('CHANNELS', 32)}, dropout={params.get('DROPOUT', 0.1)}")
        
    elif params['MODEL_NAME'] == 'BasicUNet':
        # Use MONAI's BasicUNet implementation
        model = BasicUNet(
            spatial_dims=params.get('SPATIAL_DIMS', 3),
            in_channels=params.get('IN_CHANNELS', 1),
            out_channels=params.get('OUT_CHANNELS', 1),
            act='RELU',
            norm='INSTANCE',
            dropout=params.get('DROPOUT', 0.1),
            upsample='deconv'
        )
        print(f"Created BasicUNet with features={params.get('CHANNELS', (32, 32, 64, 128, 256, 32))}, dropout={params.get('DROPOUT', 0.1)}")
        
    elif params['MODEL_NAME'] == 'AttentionUNet':
        ch = params['CHANNELS']
        sd = params['STRIDES']
        if isinstance(ch, int):
            depth = params.get('N_LAYERS', len(sd) if isinstance(sd, (list, tuple)) else 4)
            channels = [ch * (2 ** i) for i in range(depth)]
        else:
            channels = ch
        if isinstance(sd, int):
            strides = [sd] * len(channels)
        else:
            strides = sd
        model = AttentionUnet(
            spatial_dims=params['SPATIAL_DIMS'],
            in_channels=params['IN_CHANNELS'],
            out_channels=params['OUT_CHANNELS'],
            channels=channels,
            strides=strides,
            kernel_size=3,
            dropout=0.0
        )
        print(f"Created AttentionUNet with channels={channels}, strides={strides}")
        
    elif params['MODEL_NAME'] == 'WingsNet':
        model = WingsNet(
            in_channel=params.get('IN_CHANNELS', 1),
            n_classes=params.get('OUT_CHANNELS', 1)
        )
        print(f"Created WingsNet with in_channel={params.get('IN_CHANNELS', 1)}, out_channels={params.get('OUT_CHANNELS', 1)}")
        
    else:
        raise ValueError(f"Unknown model name: {params['MODEL_NAME']}. Supported models: ResUNet, SimpleResUNet, BasicUNet, AttentionUNet, WingsNet")
    
    if model is None:
        raise ValueError(f"Failed to create model: {params['MODEL_NAME']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} trainable parameters")
    
    return model


def get_loss(l_name='DiceLoss', params: Optional[dict] = None, include_background=True, to_onehot_y=False, sigmoid=True, softmax=False):
    if params is None:
        params = {}
    loss_kwargs = {
        "include_background": include_background,
        "to_onehot_y": to_onehot_y,
        "sigmoid": sigmoid,
        "softmax": softmax
    }

    match l_name:
        case 'DiceLoss':
            return DiceLoss(**loss_kwargs)
        case 'GeneralizedDiceLoss':
            print("Using GeneralizedDiceLoss")
            loss_kwargs['include_background'] = False
            return GeneralizedDiceLoss(**loss_kwargs)
        case 'DiceCELoss':
            # DiceCELoss specific params if any, e.g., lambda_dice, lambda_ce
            loss_kwargs["lambda_dice"] = params.get("lambda_dice", 1.0)
            loss_kwargs["lambda_ce"] = params.get("lambda_ce", 1.0)
            return DiceCELoss(**loss_kwargs)
        case 'IoULoss': # MONAI uses DiceLoss with jaccard=True for IoU
            loss_kwargs["jaccard"] = True
            return DiceLoss(**loss_kwargs)
        case 'MaskedDiceLoss':
            return MaskedDiceLoss(**loss_kwargs)
        case 'MaskedIoULoss': # MONAI uses MaskedDiceLoss with jaccard=True
            loss_kwargs["jaccard"] = True
            return MaskedDiceLoss(**loss_kwargs)
        case 'DiceFocalLoss':
            # DiceFocalLoss specific params, e.g., lambda_dice, lambda_focal
            loss_kwargs["lambda_dice"] = params.get("lambda_dice", 1.0)
            loss_kwargs["lambda_focal"] = params.get("lambda_focal", 1.0)
            loss_kwargs["focal_loss_gamma"] = params.get("gamma", 2.0) # Common name for gamma in FocalLoss
            return DiceFocalLoss(**loss_kwargs)
        case 'TverskyLoss':
            tversky_alpha = float(params.get('alpha', 0.3)) # Default if not in params
            tversky_beta = float(params.get('beta', 1.0 - tversky_alpha)) # Default based on alpha
            return TverskyLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid, alpha=tversky_alpha, beta=tversky_beta)
        case 'HardclDiceLoss':
            # Extract HardclDiceLoss specific params from 'params' dict if they exist
            hardcl_kwargs = loss_kwargs.copy()
            hardcl_kwargs["iter_"] = int(params.get("iter_", 3))
            hardcl_kwargs["alpha"] = float(params.get("alpha", 0.5)) # Note: 'alpha' here is for HardclDiceLoss
            return HardclDiceLoss(**hardcl_kwargs)
        case 'soft_dice_cldice':
            # Extract soft_dice_cldice specific params (iter_, alpha, smooth)
            cldice_kwargs = loss_kwargs.copy()
            cldice_kwargs["iter_"] = int(params.get("iter_", 3))
            cldice_kwargs["alpha"] = float(params.get("alpha", 0.5))
            cldice_kwargs["smooth"] = float(params.get("smooth", 1e-5))
            # This is a factory function, so call it
            return soft_dice_cldice(**cldice_kwargs)
        case 'GUL' | 'BEL': # GeneralUnionLoss
            gul_kwargs = loss_kwargs.copy()
            gul_kwargs["alpha"] = float(params.get("alpha", 0.2)) # Alpha for GUL
            gul_kwargs["rl"] = float(params.get("rl", 0.5))     # rl for GUL
            return GeneralUnionLoss(**gul_kwargs)
        case 'CBEL':
            cbel_kwargs = loss_kwargs.copy()
            # CBEL takes lists for alpha and rl
            cbel_kwargs["alpha"] = params.get("alpha", [0.2, 0.1]) 
            cbel_kwargs["rl"] = params.get("rl", [0.8, 0.5])
            return CBEL(**cbel_kwargs)
        case 'SoftclDiceLoss':
            return SoftclDiceAdapter(
                iter_=int(params.get("iter_", int(os.getenv('CLDICE_ITER', '3')))),
                smooth=float(params.get("smooth", float(os.getenv('CLDICE_SMOOTH', '1.0'))))
            )
        case 'SoftDiceclDiceLoss':
            return SoftDiceclDiceAdapter(
                iter_=int(params.get("iter_", int(os.getenv('CLDICE_ITER', '3')))),
                alpha=float(params.get("alpha", float(os.getenv('CLDICE_ALPHA', '0.5')))),
                smooth=float(params.get("smooth", float(os.getenv('CLDICE_SMOOTH', '1.0'))))
            )
        case 'DiceCEPlusSoftclDice' | 'DiceCE+SoftclDice':
            return DiceCEPlusSoftclDice(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                lambda_dice=float(params.get("lambda_dice", float(os.getenv('DICECE_LAMBDA_DICE', '1.0')))),
                lambda_ce=float(params.get("lambda_ce", float(os.getenv('DICECE_LAMBDA_CE', '0.5')))),
                cl_weight=float(os.getenv('CLDICE_WEIGHT', '0.3')),
                cl_iter=int(os.getenv('CLDICE_ITER', '3')),
                cl_smooth=float(os.getenv('CLDICE_SMOOTH', '1.0')),
            )
        case _:
            raise ValueError(f"Unknown loss function: {l_name}")


def get_optimizer(m_params, opt_name='Adam', lr=1e-3, wd=1e-5):
    match opt_name:
        case 'Adam':
            return torch.optim.Adam(m_params, lr=lr, weight_decay=wd)
        case 'AdamW':
            return torch.optim.AdamW(m_params, lr=lr, weight_decay=wd)


def get_metric(metric_name='DiceMetric', include_background=False, reduction="mean_batch"):
    match metric_name:
        case 'DiceMetric':
            return DiceMetric(include_background=include_background, reduction=reduction)
        case 'DiceMetricBatch':
            return DiceMetric(include_background=include_background, reduction=reduction)
        case 'MeanIoU':
            return MeanIoU(include_background=include_background, reduction=reduction)
        case 'LocalIoU':
            pass


def get_scheduler(optimizer, type='StepLR', with_warmup=False, step_size=15000, patience=200, factor=0.3, verbos=False, T_max=None):
    match type:
        case 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='max',
                patience=patience,
                factor=factor,
                min_lr=1e-6)
        case 'ReduceLROnPlateauWithWarmUp':
            # Todo:
            return None
        case 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
        case 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[30, 60, 90, 60],
                                                        gamma=factor)
        case 'CosineAnnealing':
            # Use T_max if provided, otherwise fall back to step_size
            t_max = T_max if T_max is not None else step_size
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=t_max,
                                                        eta_min=1e-6)


def save_val_2path(output_p: str, type_p: str, f_name: str, tensor: torch.Tensor, affine: np.array, header: np.array) -> None:
    # 1.  Ensure we have a NumPy array
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()

    # 2.  Cast to a safe dtype
    if tensor.dtype == np.int64:
        # binary or label volumes → uint8
        tensor = tensor.astype(np.uint8)
    elif tensor.dtype == np.float64:
        # probability maps → float32
        tensor = tensor.astype(np.float32)

    # 3.  Make sure the header advertises the right dtype
    if header is not None:
        header = header.copy()
        header.set_data_dtype(tensor.dtype)

    nifti_img = nib.Nifti1Image(tensor, affine, header)
    f_path = os.path.join(output_p, (type_p or ""), f"{f_name}.nii.gz")
    print(f"Saving to {f_path} …", end="")
    nib.save(nifti_img, f_path)
    print("done.")


def get_lcc(pred: np.ndarray) -> np.ndarray:
    """
    Keep the largest 3-D connected component and fill holes.
    Works for np.ndarray or torch.Tensor with shape [Z,Y,X] (or [1,Z,Y,X]).
    """
    # to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    vol = np.squeeze(pred) > 0          # [Z,Y,X] boolean
    if vol.ndim != 3:
        raise ValueError(f"get_lcc expects a 3-D mask, got {vol.shape}")

    # Add batch dimension so MONAI treats it as one 3-D sample
    # and uses 3-D connectivity (connectivity defaults to ndim=3).
    vol_batched = vol[None, ...]        # [1,Z,Y,X]
    mask_batched = get_largest_connected_component_mask(vol_batched, connectivity=3)
    lcc = np.asarray(mask_batched)[0]   # back to [Z,Y,X]

    if os.getenv("FILL_HOLES", "True").lower() == "true":
        lcc = binary_fill_holes(lcc).astype(np.uint8)
    else:
        lcc = lcc.astype(np.uint8)
    return lcc


def remove_small_components(pred: np.ndarray, min_size: int = 64) -> np.ndarray:
    """
    Remove connected components smaller than a specified size.

    Args:
        pred: The binary prediction mask as a NumPy array.
        min_size: The minimum size of components to keep.

    Returns:
        The mask with small components removed.
    """
    # Ensure pred is a numpy array and squeeze it
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    pred = np.squeeze(pred)

    if pred.ndim != 3:
        raise ValueError(f"Expected a 3D array after squeezing, but got shape {pred.shape}.")

    # Label connected components
    labeled_pred, num_components = measure.label(pred > 0, return_num=True, connectivity=1)
    
    if num_components == 0:
        return np.zeros_like(pred, dtype=np.uint8)

    # Calculate volumes of all components
    component_volumes = np.bincount(labeled_pred.ravel())
    
    # Identify labels of components that are too small
    # component_volumes[0] is the background, so we ignore it.
    small_components = np.where(component_volumes[1:] < min_size)[0] + 1
    
    # If all components are small, return an empty mask
    if len(small_components) == num_components:
        return np.zeros_like(pred, dtype=np.uint8)

    # Create a mask of the small components to be removed
    remove_mask = np.isin(labeled_pred, small_components)
    
    # Subtract the small components from the original prediction
    pred[remove_mask] = 0
    
    return pred.astype(np.uint8)

def save_one_hot_encoding(tensor, save_path):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    # Save the tensor to the specified path
    np.save(save_path, tensor)
    print(f"Tensor saved to {save_path} successfully.")


def find_file_in_path(directory, filename):
    """
    Searches for a file with the specified filename in the given directory.

    Parameters:
    directory (str): The path to the directory to search in.
    filename (str): The name of the file to find.

    Returns:
    str: The full path to the file if found, otherwise None.
    """
    for root, dirs, files in os.walk(directory):
        # for file in files:
        #     print(file)
        # print(filename)
        if filename+".nii.gz" in files:
            return str(os.path.join(root, filename+".nii.gz"))
    return None


def get_folder_names(dataset):
    """
    Returns folder names based on the dataset type.

    Parameters:
    dataset (str): The type of dataset ("AIIB23", "ATM22", or "AeroPath").

    Returns:
    tuple: A tuple containing the image path, labels path, and lungs path.
    """
    if dataset == "AIIB23":
        return 'img/', 'gt/', 'lungs/'
    elif dataset == "ATM22":
        return 'imagesTr/', 'labelsTr/', 'lungsTr/'
    elif dataset == "AeroPath":
        return 'CT/', 'airway/', 'lungs/'
    else:
        raise ValueError("Unsupported dataset type. Choose from 'AIIB23', 'ATM22', or 'AeroPath'.")

def load_patch_bank_for_case(case_id: str, patch_banks_dir: str | Path):
    """Return (patch_list, meta_dict) or (None, None) if missing."""
    bank_path = Path(patch_banks_dir) / f"{case_id}.pkl.gz"
    if not bank_path.exists():
        return None, None
    with gzip.open(str(bank_path), "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "patch_descriptors" in obj:
        return obj["patch_descriptors"], obj.get("metadata", {})
    return obj, {}

def extract_case_id_from_meta(batch_meta: dict) -> str:
    """get the case id from MONAI meta."""
    v = None
    if isinstance(batch_meta, dict):
        v = batch_meta.get("filename_or_obj")
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
    if isinstance(v, (str, Path)):
        return Path(v).stem
    return "unknown"

def _sliding_window_starts(shape_zyx: Tuple[int, int, int],
                           roi_zyx: Tuple[int, int, int],
                           overlap: float) -> List[Tuple[int, int, int]]:
    # step per dim per MONAI logic: floor(roi_size * (1 - overlap)), at least 1 voxel
    steps = [max(1, int(r * (1.0 - float(overlap)))) for r in roi_zyx]
    ranges = []
    for dim, r, st in zip(shape_zyx, roi_zyx, steps):
        last = max(0, dim - r)
        idxs = list(range(0, max(1, dim - r + 1), st))
        if not idxs or idxs[-1] != last:
            idxs.append(last)
        ranges.append(idxs)
    starts = [(z, y, x) for z in ranges[0] for y in ranges[1] for x in ranges[2]]
    return starts

def _gaussian_importance_map(roi_zyx: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Simple separable Gaussian window centered in ROI, max=1.0 (like MONAI's)."""
    z, y, x = [int(v) for v in roi_zyx]
    def g(n):
        t = torch.linspace(-1, 1, n, device=device)
        return torch.exp(-0.5 * (t / 0.3) ** 2)  # sigma ~ 0.3 of half-width
    wz, wy, wx = g(z), g(y), g(x)
    win = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    win /= win.max()
    return win  # (Z,Y,X)

def make_audit_weight_provider(patch_list: List[Dict],
                               roi_zyx: Tuple[int, int, int]) -> Callable[[Tuple[int,int,int]], float]:
    """Return f(start_zyx)-> normalized weight using nearest audited patch."""
    positions = np.array([p.get("position_zyx", p.get("position", (0,0,0))) for p in patch_list],
                         dtype=np.int32)
    weights = np.array([float(p.get("sampling_weight", 1.0)) for p in patch_list], dtype=np.float32)
    if len(positions) == 0:
        return lambda *_: 1.0
    if cKDTree is not None:
        tree = cKDTree(positions)
        def _provider(start):
            _, idx = tree.query(np.asarray(start), k=1)
            return float(weights[int(idx)])
    else:
        def _provider(start):
            d2 = ((positions - np.asarray(start)) ** 2).sum(axis=1)
            idx = int(np.argmin(d2))
            return float(weights[idx])
    mean_w = float(weights.mean()) if weights.size else 1.0
    mean_w = 1.0 if mean_w <= 0 else mean_w
    return lambda s: _provider(s) / mean_w

def make_landmark_weight_provider_from_bank(patch_list: List[Dict],
                                            vol_shape_zyx: Tuple[int,int,int],
                                            roi_zyx: Tuple[int,int,int],
                                            spacing_zyx_mm: Optional[Tuple[float,float,float]] = None,
                                            sigma_mm: float = 30.0) -> Callable[[Tuple[int,int,int]], float]:
    """
    Seed trachea/bifurcation centers from bank flags, build distance maps,
    return w(start) = exp(-min(D_trach, D_bif)/sigma) normalized by case mean.
    """
    dz, dy, dx = [int(v) for v in roi_zyx]
    # collect patch centers for flagged descriptors
    tr_centers, bif_centers = [], []
    for p in patch_list:
        z, y, x = [int(v) for v in p.get("position_zyx", p.get("position", (0,0,0)))]
        cz, cy, cx = z + dz//2, y + dy//2, x + dx//2
        if bool(p.get("is_trachea", False)):
            tr_centers.append((cz, cy, cx))
        if bool(p.get("is_bifurcation", False)):
            bif_centers.append((cz, cy, cx))
    if not tr_centers and not bif_centers:
        return lambda *_: 1.0  # no landmark info → uniform

    seeds_tr = np.zeros(vol_shape_zyx, dtype=np.uint8)
    seeds_bf = np.zeros(vol_shape_zyx, dtype=np.uint8)
    for (cz, cy, cx) in tr_centers:
        if 0 <= cz < vol_shape_zyx[0] and 0 <= cy < vol_shape_zyx[1] and 0 <= cx < vol_shape_zyx[2]:
            seeds_tr[cz, cy, cx] = 1
    for (cz, cy, cx) in bif_centers:
        if 0 <= cz < vol_shape_zyx[0] and 0 <= cy < vol_shape_zyx[1] and 0 <= cx < vol_shape_zyx[2]:
            seeds_bf[cz, cy, cx] = 1

    # Distance in mm if spacing is known; otherwise in voxels.
    sampling = spacing_zyx_mm if (spacing_zyx_mm is not None) else None
    Dtr = distance_transform_edt(1 - seeds_tr, sampling=sampling)
    Dbf = distance_transform_edt(1 - seeds_bf, sampling=sampling)
    Dmin = np.minimum(Dtr, Dbf)

    if spacing_zyx_mm is None:
        sigma = float(sigma_mm)  # interpret as "voxels" if spacing unknown
    else:
        # use mean spacing to keep it simple; sigma_mm is isotropic
        sigma = float(sigma_mm)

    def raw_w(start):
        # weight at window center
        cz, cy, cx = start[0] + dz//2, start[1] + dy//2, start[2] + dx//2
        cz = min(max(cz, 0), vol_shape_zyx[0]-1)
        cy = min(max(cy, 0), vol_shape_zyx[1]-1)
        cx = min(max(cx, 0), vol_shape_zyx[2]-1)
        return float(np.exp(- Dmin[cz, cy, cx] / (sigma + 1e-8)))
    # normalize by mean over all starts (computed lazily by caller for efficiency)
    return raw_w

# ---- main: weighted sliding-window inference --------------------------------
@torch.no_grad()
def weighted_sliding_window_inference(
    inputs: torch.Tensor,                     # [B=1, C, Z, Y, X]
    predictor,                                # model.forward
    roi_size: Tuple[int,int,int],
    overlap: float,
    per_window_weight: Callable[[Tuple[int,int,int]], float],
    device: Optional[torch.device] = None,
    gaussian_blend: bool = True,
) -> torch.Tensor:
    """
    Inspired from MONAI's function but adds a scalar per-window weight.
    Accumulates: sum_logits += w_i * G * logits_i; sum_w += w_i * G.
    """
    assert inputs.ndim == 5 and inputs.shape[0] == 1, "supports B=1 for simplicity"
    device = device or inputs.device
    _, _, Z, Y, X = inputs.shape
    roi_zyx = tuple(int(v) for v in roi_size)
    starts = _sliding_window_starts((Z, Y, X), roi_zyx, overlap)

    imp = _gaussian_importance_map(roi_zyx, device) if gaussian_blend else torch.ones(roi_zyx, device=device)
    imp = imp.clamp_min(1e-6)[None, None, ...]  # [1,1,z,y,x]

    sum_logits = torch.zeros((1, 1, Z, Y, X), dtype=torch.float32, device=device)
    sum_w = torch.zeros_like(sum_logits)

    # To normalize w_i by case mean, collect once:
    with torch.no_grad():
        w_list = [float(per_window_weight(s)) for s in starts]
    mean_w = float(np.mean(w_list)) if len(w_list) else 1.0
    mean_w = 1.0 if mean_w <= 0 else mean_w

    for s, w_raw in zip(starts, w_list):
        z, y, x = s
        z2, y2, x2 = z + roi_zyx[0], y + roi_zyx[1], x + roi_zyx[2]
        tile = inputs[..., z:z2, y:y2, x:x2]  # [1,C,rz,ry,rx]
        logits = predictor(tile)              # [1,1,rz,ry,rx] (assume 1‑ch out)

        w = float(w_raw) / mean_w             # case‑normalized
        wmap = (w * imp).to(logits.dtype)     # [1,1,rz,ry,rx]

        sum_logits[..., z:z2, y:y2, x:x2] += wmap * logits
        sum_w[..., z:z2, y:y2, x:x2] += wmap

    out = sum_logits / (sum_w + 1e-6)
    return out  # logits