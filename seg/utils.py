import math
import os
from typing import Optional
import numpy as np
import torch
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, MaskedDiceLoss, DiceFocalLoss, GeneralizedDiceLoss
from .loss_functions import HardclDiceLoss, GeneralUnionLoss, soft_dice_cldice, CBEL
try:
    # Use absolute import instead of relative import
    from models.WingsNet import WingsNet
    from models.ResUNet import ResUNet
except ImportError:
    # Fallback with improved path handling
    import sys
    import os.path as path
    parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from models.WingsNet import WingsNet
    from models.ResUNet import ResUNet
from monai.metrics import DiceMetric, MeanIoU
from torch import nn
from monai.networks.nets import BasicUNet, UNet, AttentionUnet
from monai.utils import set_determinism, MetricReduction, Method
import random
import pickle
import sys
import nibabel as nib
from monai.transforms import SpatialResample, get_largest_connected_component_mask

random.seed(1221)
np.random.seed(1221)
# safer determinism setting to avoid CUDA errors
try:
    # Set CUBLAS workspace config for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    # Allow non-deterministic algorithms when deterministic implementation is not available
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(seed=0)
    print("Utils: Using deterministic algorithms with fallback for CUDA compatibility")
except Exception as e:
    # Fallback: set seeds but allow non-deterministic algorithms
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    print(f"Utils: Warning - Using non-deterministic algorithms for CUDA compatibility: {e}")


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
        # Use the new ResUNet implementation with attention gates
        from monai.networks.layers.factories import Act, Norm
        
        # Convert string norm to enum if needed
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
            num_res_units=params.get('NUM_RES_UNITS', 1),  # Not used in new implementation but kept for compatibility
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
            dropout=params['DROPOUT']
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

    # Extract common MONAI loss params, allow overrides from specific loss logic if needed
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
            # Recommended for imbalanced data, focuses on foreground
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
    Get the largest connected component from a binary mask using MONAI.
    
    Args:
        pred: The binary prediction mask as a NumPy array.
        
    Returns:
        The largest connected component of the mask.
    """
    # Ensure pred is a numpy array
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # Ensure we pass a 3-D volume to skimage (it only supports 1-3 D).
    # If a singleton leading axis exists, drop it; otherwise keep as-is.
    while pred.ndim > 3 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)

    if pred.ndim > 3:
        raise ValueError(
            f"get_lcc expects <=3-D mask per sample, got shape {pred.shape}."
        )

    lcc_mask = get_largest_connected_component_mask(pred)

    return lcc_mask.astype(np.uint8)


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


# def resample_and_save_nifti(data_to_save: np.ndarray,
#                               source_affine: np.ndarray,
#                               target_affine: np.ndarray,
#                               target_shape: tuple,
#                               output_filename: str,
#                               mode: str = 'nearest',
#                               original_header=None):
#     """
#     Resamples a NumPy array to a target space defined by target_affine and target_shape,
#     then saves it as a NIfTI image.

#     Args:
#         data_to_save: The NumPy array data to resample and save.
#         source_affine: The affine matrix of the data_to_save.
#         target_affine: The desired affine matrix for the output NIfTI file.
#         target_shape: The desired spatial shape for the output NIfTI file.
#         output_filename: Path to save the resampled NIfTI image.
#         mode: Interpolation mode ('nearest', 'bilinear', 'trilinear', etc.).
#         original_header: Optional original NIfTI header to use for the new file.
#                          If provided, its data dtype will be updated.
#     """
#     if not isinstance(data_to_save, np.ndarray):
#         raise ValueError("data_to_save must be a NumPy array.")

#     # Ensure data is float32 for resampling, can be cast later
#     img_tensor = torch.tensor(data_to_save.astype(np.float32)).unsqueeze(0)  # Add batch dim (B,H,W,D)
#     if img_tensor.ndim == 4: # if it's (B, H, W, D), needs to be (B, C, H, W, D) for MONAI
#         img_tensor = img_tensor.unsqueeze(1) # Add channel dim -> (B, C, H, W, D)
#     elif img_tensor.ndim == 5: # Already (B,C,H,W,D)
#         pass
#     else:
#         raise ValueError(f"Input data has unexpected ndim: {img_tensor.ndim}")

#     # Create a Resample transform instance
#     # We need to provide MONAI with the source image's affine.
#     # MONAI's Resample transform by default assumes the input tensor's affine is identity.
#     # To correctly resample from a known source affine to a target affine/shape,
#     # we need to ensure the input tensor carries its affine information or the transform
#     # is configured to understand it.
#     # A common way is to create a MetaTensor.
#     from monai.data import MetaTensor
#     # Create a MetaTensor from the input tensor and its source affine
#     # The Resample transform will use this source affine.
#     img_metatensor = MetaTensor(img_tensor, affine=torch.tensor(source_affine, dtype=torch.float32))


#     # Create a dummy MetaTensor that defines the target space (affine and shape)
#     # This is not strictly necessary if dst_affine and spatial_size are used directly,
#     # but 'like' can be more robust.
#     # However, for simplicity and directness, we'll use dst_affine and spatial_size.
    
#     resampler = SpatialResample(mode=mode, padding_mode='zeros', align_corners=False, dtype=img_metatensor.dtype)

#     # Resample the image.
#     # The Resample transform needs the target affine and shape.
#     # It's important that the target_affine is compatible with the target_shape.
#     # The `dst_affine` parameter in Resample is what we need.
#     # It will resample the `img_metatensor` (which knows its own `source_affine`)
#     # to the space defined by `dst_affine` and `spatial_size`.
    
#     # Ensure target_affine is a tensor
#     target_affine_tensor = torch.tensor(target_affine, dtype=torch.float32)

#     # Ensure target_shape is a tuple of ints
#     if isinstance(target_shape, torch.Size):
#         target_shape_tuple = tuple(target_shape)
#     elif isinstance(target_shape, (list, np.ndarray)):
#         target_shape_tuple = tuple(int(s) for s in target_shape)
#     elif isinstance(target_shape, tuple):
#         target_shape_tuple = tuple(int(s) for s in target_shape)
#     else:
#         raise ValueError(f"target_shape must be a tuple, list, np.ndarray, or torch.Size, got {type(target_shape)}")

#     # Pass img_metatensor[0] (C,H,W,D) to the resampler because the functional version
#     # being used expects (C,H,W,D) and applies an unconditional unsqueeze(0).
#     resampled_chwd = resampler(img_metatensor[0], dst_affine=target_affine_tensor, spatial_size=target_shape_tuple)
#     # Restore the batch dimension.
#     resampled_tensor = resampled_chwd.unsqueeze(0)
    
#     resampled_np = resampled_tensor.cpu().numpy().squeeze() # Remove Batch and Channel if they are 1

#     # Cast to appropriate dtype after resampling
#     if mode == 'nearest': # Typically for segmentations
#         resampled_np = resampled_np.astype(np.uint8)
#     else: # Typically for probabilities
#         resampled_np = resampled_np.astype(np.float32)

#     # # Prepare header
#     # header_to_save = None
#     # if original_header is not None:
#     #     header_to_save = original_header.copy()
#     #     header_to_save.set_data_dtype(resampled_np.dtype)
#     #     # Update shape in header if possible (some headers might not store it directly like this)
#     #     header_to_save.set_data_shape(resampled_np.shape)


#     # # Create NIfTI image with the target_affine
#     # nifti_img = nib.Nifti1Image(resampled_np, target_affine, header_to_save)
    
#     # # Ensure directory exists
#     # os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
#     # print(f"Resampling to shape {target_shape} and saving to {output_filename} ...", end="")
#     # nib.save(nifti_img, output_filename)
#     # print("done.")
#     # Prepare header
#     header_to_save = None
#     if original_header is not None:
#         header_to_save = original_header.copy()
#         header_to_save.set_data_dtype(resampled_np.dtype)
#         # Update shape in header if possible (some headers might not store it directly like this)
#         header_to_save.set_data_shape(resampled_np.shape)

#     # Create NIfTI image with the target_affine - no orientation transform applied
#     os.makedirs(os.path.dirname(output_filename), exist_ok=True)
#     print(f"Resampling to shape {target_shape} and saving to {output_filename} ...", end="")
#     nifti_img = nib.Nifti1Image(resampled_np, target_affine, header_to_save)
#     nib.save(nifti_img, output_filename)
#     print("done.")

