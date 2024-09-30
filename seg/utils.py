import math
import os
import numpy as np
import torch
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, MaskedDiceLoss, DiceFocalLoss
from loss_functions import HardclDiceLoss, GeneralUnionLoss, soft_dice_cldice, CBEL
from monai.metrics import DiceMetric, MeanIoU
from torch import nn
from monai.networks.nets import BasicUNet, UNet, AttentionUnet
from monai.utils import set_determinism, MetricReduction, Method
import random
import pickle
import sys
import nibabel as nib
import skimage.measure as measure
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation

random.seed(1221)
np.random.seed(1221)
set_determinism(seed=0)


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
    output = model(model_input.to(device)).sum()
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
    co = int(math.log(params['CHANNELS'], 2))
    channels = tuple((2 ** j) for j in range(co, co + params['N_LAYERS']))
    # if params['N_LAYERS'] > 5:
    #     pass
    strides = ((params['STRIDES']),) * params['N_LAYERS']
    match params['MODEL_NAME']:
        case 'ResUNet':
            model = UNet(
                spatial_dims=params['SPATIAL_DIMS'],
                in_channels=params['IN_CHANNELS'],
                out_channels=params['OUT_CHANNELS'],
                channels=channels,
                strides=strides,
                num_res_units=params['NUM_RES_UNITS'],
                dropout=params['DROPOUT'],
            )
        case 'AttentionUNet':
            model = AttentionUnet(
                spatial_dims=params['SPATIAL_DIMS'],
                in_channels=params['IN_CHANNELS'],
                out_channels=params['OUT_CHANNELS'],
                channels=channels,
                strides=strides,
                kernel_size=[3, 3, 3],
                up_kernel_size=[3, 3, 3],
                dropout=params['DROPOUT'],
            )
        case 'BasicUNet':
            pass
        case 'WingsNet':
            model = WingsNet()
    return model


def get_loss(l_name='DiceLoss', include_background=True, to_onehot_y=False, sigmoid=False, softmax=False, alpha=None, rl=None):
    match l_name:
        case 'DiceLoss':
            return DiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        case 'DiceCELoss':
            return DiceCELoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        case 'IoULoss':
            return DiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid,
                            jaccard=True)
        case 'MaskedDiceLoss':
            return MaskedDiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        case 'MaskedIoULoss':
            return MaskedDiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid,
                                  jaccard=True)
        case 'DiceFocalLoss':
            return DiceFocalLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        case 'TverskyLoss':
            return TverskyLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid,
                               alpha=0.1, beta=0.9)
        case 'HardclDiceLoss':
            return HardclDiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid)
        case 'soft_dice_cldice':
            return soft_dice_cldice(include_background=include_background, to_onehot_y=to_onehot_y, sigmoid=sigmoid,
                                    # iter_=3, alpha=0.5, smooth = 1e-5
                                    )
        case 'GUL':
            return GeneralUnionLoss(include_background=include_background, alpha=alpha, rl=rl, sigmoid=sigmoid)
        case 'BEL':
            return GeneralUnionLoss(include_background=include_background, alpha=alpha, rl=rl, sigmoid=sigmoid)
        case 'CBEL':
            return CBEL(include_background=include_background, alpha=[0.2, 0.1], rl=[0.8, 0.5], sigmoid=sigmoid, softmax=softmax)


def get_optimizer(m_params, opt_name='Adam', lr=1e-3, wd=1e-5):
    match opt_name:
        case 'Adam':
            return torch.optim.Adam(m_params, lr=lr, weight_decay=wd)
        case 'AdamW':
            return torch.optim.AdamW(m_params, lr=lr, weight_decay=wd)


def get_metric(metric_name='DiceMetric', include_background=True, reduction="mean"):
    match metric_name:
        case 'DiceMetric':
            return DiceMetric(include_background=include_background, reduction=reduction)
        case 'DiceMetricBatch':
            return DiceMetric(include_background=include_background, reduction=reduction)
        case 'MeanIoU':
            return MeanIoU(include_background=include_background, reduction=reduction)
        case 'LocalIoU':
            pass


def get_scheduler(optimizer, type='StepLR', with_warmpup=False, step_size=15000, patience=200, factor=0.1, verbos=False):
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


def save_val_2path(output_p: str, type_p: str, f_name: str, tensor: torch.Tensor, affine: np.array, header: np.array) -> None:
    nifti_img = nib.Nifti1Image(tensor, affine, header)
    f_path = output_p + f"/{type_p}/{f_name}.nii.gz"
    print(f"Saving to {f_path} ...", end="")
    nib.save(nifti_img, f_path)
    print("Done!")


def get_lcc(pred):
    cd, num = measure.label(pred, return_num=True, connectivity=1)
    print(f"\n Number of connected components is {num}")
    if num < 1:
        print("No calculations... \n")
        return 0, 0, 0, 0, 0, 0
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    return (cd == (volume_sort[-1] + 1)).astype(np.uint8)

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

