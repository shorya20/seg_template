import os
import argparse
import json
import logging
from pathlib import Path
import torch.nn as nn
import sys
from typing import Any, Optional
from monai.utils import first
import numpy as np
import time
import random
import datetime
import nibabel as nib
import monai
import copy as cp
from callbacks import SmartCacheCallback
from .force_gpu import (force_cuda_device, diagnose_model_placement, verify_tensor_on_gpu, force_model_to_gpu, monitor_gpu_usage)
from .utils import (
    weighted_sliding_window_inference, load_patch_bank_for_case,
    extract_case_id_from_meta, make_audit_weight_provider,
    make_landmark_weight_provider_from_bank, get_model, get_loss, get_optimizer, get_scheduler, save_to_check, test_memory_inference, test_memory_training,
    get_lcc, save_val_2path, get_folder_names, find_file_in_path, save_one_hot_encoding, get_metric
)
from .DataModule import SegmentationDataModule
from .metrics import AirwayMetrics
from .topo_metrics import compute_binary_iou, evaluation_branch_metrics
import json
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.data.meta_tensor import MetaTensor
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    AsDiscrete,
    Rotate90,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    Invertd,
    AsDiscreted,
    Activationsd,
    KeepLargestConnectedComponentd,
    Resized,
    Resize,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    NormalizeIntensityd,
    EnsureType,
    Activations,
    AsDiscrete,
    KeepLargestConnectedComponent,
    VoteEnsembled,
    DivisiblePadd,
    SaveImaged,
    CastToTyped,
    MapLabelValued,
    Lambdad
)
from monai.optimizers import WarmupCosineSchedule
from dotenv import load_dotenv, find_dotenv
import csv
import resource
import gc
import torch.amp as amp
import resource
import traceback
import math
from monai.metrics import DiceMetric, MeanIoU
import multiprocessing as mp
import re
from tqdm import tqdm
from scipy.ndimage import zoom
import scipy.ndimage
from scipy.ndimage import binary_erosion, label as ndimg_label
from models.SimpleResUNet import SimpleResUNet
from monai.transforms.transform import MapTransform
from .transforms import DebugMetadatad
from .custom_save import SaveImageWithOriginalMetadatady
import torch.serialization
from monai.utils.enums import LazyAttr
from .DataModule import LoadOriginalNiftiHeaderd
torch.serialization.add_safe_globals([LazyAttr])
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    if weights_only is None:
        # Check if this looks like a MONAI cache file
        if hasattr(f, 'name') and isinstance(f.name, str):
            # PersistentDataset cache files are typically in cache directories
            if 'cache' in f.name.lower() or f.name.endswith('.pt'):
                weights_only = False
        elif isinstance(f, str):
            if 'cache' in f.lower() or f.endswith('.pt'):
                weights_only = False
    
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=weights_only, **kwargs)

# Apply the monkey patch
torch.load = _patched_torch_load

try:
    # 'spawn' is the safest choice with CUDA; avoids deadlocks seen with 'fork'
    mp.set_start_method('spawn', force=True)
    print("Set multiprocessing start method to 'spawn'")
except RuntimeError:
    # already set
    current = mp.get_start_method(allow_none=True)
    print(f"Multiprocessing start method already set to '{current}'")

# Get current limits
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
torch.backends.cudnn.benchmark = False
# more reasonable value or use the existing hard limit
try:
    new_soft_limit = min(131072, rlimit[1])
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, rlimit[1]))
except ValueError:
    # If that fails, just keep the current limits
    print(f"Warning: Could not increase file limit. Using current limit: {rlimit[0]}")
torch.set_float32_matmul_precision('high')

pl.seed_everything(42, workers=True)

def parse_args():
    parser = argparse.ArgumentParser(description="The airway segmentation on ATM22")
    parser.add_argument("--max_time_hours", type=int, default=int(os.getenv('MAX_TIME_HOURS', '120')))
    # patch size - for now static
    parser.add_argument("--patch_size", type=int, default=int(os.getenv('PATCH_SIZE', 192)), help="Minimum pixel value")
    parser.add_argument("--test_patch_size", type=int, default=None, help="Specific patch size for testing phase (defaults to patch_size if not set)")
    parser.add_argument("--debug_inference", action="store_true", help="Save debug images during inference to track where artifacts are introduced")
    parser.add_argument("--one_hot", action="store_true", help="run on one-hot encoding")
    parser.add_argument("--mixed", action="store_true", help="run with mixed precision training")
    # Voxel values - for now static
    parser.add_argument("--pixel_value_min", type=int, default=float(os.getenv('PIXEL_VALUE_MIN', '-1000')), help="Minimum pixel value")
    parser.add_argument("--pixel_value_max", type=int, default=float(os.getenv('PIXEL_VALUE_MAX', '600')), help="Maximum pixel value")
    parser.add_argument("--pixel_norm_min", type=float, default=float(os.getenv('PIXEL_NORM_MIN', '0.0')), help="Minimum normalized pixel value")
    parser.add_argument("--pixel_norm_max", type=float, default=float(os.getenv('PIXEL_NORM_MAX', '1.0')), help="Maximum normalized pixel value")
    # Model Hyperparameters - static for the moment
    parser.add_argument("--model_name", type=str, default=os.getenv('MODEL_NAME', 'BasicUNet'), help="Model name")
    parser.add_argument("--spatial_dims", type=int, default=int(os.getenv('SPATIAL_DIMS', '3')), help="Spatial dimensions")
    parser.add_argument("--in_channels", type=int, default=int(os.getenv('IN_CHANNELS', '1')), help="Input channels")
    parser.add_argument("--out_channels", type=int, default=int(os.getenv('OUT_CHANNELS', '1')), help="Output channels")
    parser.add_argument("--n_layers", type=int, default=int(os.getenv('N_LAYERS', '4')), help="Number of layers")
    parser.add_argument("--channels", type=int, default=int(os.getenv('CHANNELS', '16')), help="Channels")
    parser.add_argument("--strides", type=int, default=int(os.getenv('STRIDES', '2')), help="Strides")
    parser.add_argument("--num_res_units", type=int, default=int(os.getenv('NUM_RES_UNITS', '2')), help="Number of residual units")
    parser.add_argument("--dropout", type=float, default=float(os.getenv('DROPOUT', '0.2')), help="Dropout rate")
    parser.add_argument("--norm", type=str, default=os.getenv('NORM', 'INSTANCE'), help="Normalization type")

    # Training Hyperparameters
    parser.add_argument("--loss_func", type=str, default=os.getenv('LOSS_FUNC', 'DiceLoss'), help="The loss function to use")
    parser.add_argument("--alpha", type=float, default=float(os.getenv('ALPHA_VALUE', '0.3')), help="The alpha value")
    parser.add_argument("--r_d", type=float, default=1., help="The decay pattern value")
    parser.add_argument("--r_l", type=float, default=0.7, help="The power in GUL!")
    parser.add_argument("--scheduler", type=str, default=os.getenv('SCHEDULER_TYPE', 'ReduceLROnPlateau'), help="Type of scheduler")
    parser.add_argument("--patience", type=int, default=int(os.getenv('PATIENCE', '3')), help="The learning rate patience")
    parser.add_argument("--optimizer", type=str, default=os.getenv('OPTIMIZER', 'AdamW'), help="The optimizer to use")
    parser.add_argument("--lr", type=float, default=float(os.getenv('LR', '0.0003')), help="The learning rate")
    parser.add_argument("--max_epochs", type=int, default=int(os.getenv('NUM_EPOCHS', '100')), help="how many epochs?")
    parser.add_argument("--max_cardinality", type=int, default=int(os.getenv('MAX_CARDINALITY', '200')), help="size of dataset")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv('BATCH_SIZE', '1')), help="size of dataset")
    parser.add_argument("--num_workers", type=int, default=int(os.getenv('NUM_WORKERS', '4')), help="Number of DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=int(os.getenv('PREFETCH_FACTOR', '2')), help="Prefetch factor for DataLoader")
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=bool(os.getenv('GRADIENT_CHECKPOINTING', False)), 
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--smartcache", action="store_true", default=bool(os.getenv('SMARTCACHE', 'False').lower() == 'true'), 
                        help="Use SmartCacheDataset instead of regular Dataset")
    parser.add_argument("--cache_rate", type=float, default=float(os.getenv('CACHE_RATE', '0.4')), 
                        help="Cache rate for SmartCacheDataset")
    parser.add_argument("--smartcache_replace_rate", type=float, default=float(os.getenv('SMARTCACHE_REPLACE_RATE', '0.25')), 
                        help="Replace rate for SmartCacheDataset")
    parser.add_argument("--smartcache_init_workers_factor", type=float, default=float(os.getenv('SMARTCACHE_INIT_WORKERS_FACTOR', '0.5')), 
                        help="Factor to calculate number of init workers for SmartCacheDataset")
    parser.add_argument("--smartcache_replace_workers_factor", type=float, default=float(os.getenv('SMARTCACHE_REPLACE_WORKERS_FACTOR', '0.5')), 
                        help="Factor to calculate number of replace workers for SmartCacheDataset")

    # File paths
    parser.add_argument("--exp_path", type=str, default="", help="The experiment path")
    parser.add_argument("--weights_path", type=str, default="", help="The best model weight path")
    parser.add_argument("--dataset", type=str, default="ATM22", help="The DATASET to use")
    parser.add_argument("--weights_dir", type=str, default="Wr1alpha0.2", help="The weight map of mgul path")

    # Adaptive weight change AWC
    parser.add_argument("--awc", action="store_true", default=False, help="activate AWC or not")
    parser.add_argument("--awc_factor", type=float, default=1.2, help="The learning rate")

    # Other options
    parser.add_argument("--amp", action="store_true", default=False, help="For improving the training")
    parser.add_argument("--cache_dataset", action="store_true", default=False, help="Caching dataset or not")
    parser.add_argument("--save_val", action="store_true", default=False, help="Caching dataset or not")
    parser.add_argument("--print_val_stats", action="store_true", default=False, help="Print additional validation statistics")
    parser.add_argument("--pred_threshold", type=float, default=float(os.getenv('PRED_THRESHOLD', '0.5')), help="Probability threshold for binarizing predictions")

    # Phase of the experiment
    parser.add_argument("--training", action="store_true", help="Run training phase")
    parser.add_argument("--validation", action="store_true", help="Run validation phase")
    parser.add_argument("--testing", action="store_true", help="Run inference to generate predictions.")
    parser.add_argument("--metric_testing", action="store_true", help="Run testing on a labeled dataset to calculate metrics.")
    parser.add_argument("--test_dataset", type=str, default=os.getenv('TEST_DATASET', 'off_val'), choices=['off_val','val'], help="Which dataset to use for test stage: off_val (official val without GT) or val (validation split of training with GT)")
    parser.add_argument("--awc_val", action="store_true", help="Run awc validation phase")
    
    # Resume training option
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to checkpoint to resume training from")
    # Fine-tune from checkpoint (weights-only)
    parser.add_argument(
        "--finetune_from",
        type=str,
        default="",
        help="Path to checkpoint to fine-tune from (loads weights only, re-inits optimizer/scheduler)",
    )
    args = parser.parse_args()
    args.use_pretrained = args.weights_path != ""
    args.one_hot = False
    args.mixed = True
    args.weights_dir = "Wr1alpha0.2" 
    args.max_cardinality = 120 if args.dataset == "AIIB23" else 200
    if args.dataset == "AeroPath":
        args.max_cardinality = 27
    args.out_channels = 1  # use 1 output channel for binary segmentation
    return args

load_dotenv(find_dotenv("config.env"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

now = datetime.datetime.now()
day = now.day
hour = now.hour
month = now.month

# fixes the max_pool3d_with_indices_backward_cuda deterministic error
try:
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)    
    # MONAI's set_determinism - this will set seeds and some deterministic behaviors
    set_determinism(seed=42)
    print("Using deterministic algorithms with fallback for non-deterministic CUDA operations")
    
except Exception as e:
    print(f"Warning: Could not enable full determinism: {e}")
    # Fall back to setting just the random seeds without strict deterministic algorithms
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # Explicitly disable deterministic algorithms to avoid CUDA errors
    torch.use_deterministic_algorithms(False)
    print("Using non-deterministic algorithms for compatibility with CUDA operations")

# parser inputs
args = parse_args()
print(f"cache_dataset parameter value: {args.cache_dataset}")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    # 'PATCH_SIZE': (args.patch_size, args.patch_size, args.patch_size),
    # 'PATCH_SIZE': (384, 256, 512),
# Parameters
params_dict = {
    'PATCH_SIZE': (args.patch_size, args.patch_size, args.patch_size),
    'BATCH_SIZE': args.batch_size,
    'MAX_CARDINALITY': args.max_cardinality,
    'NUM_WORKERS': args.num_workers,
    'PREFETCH_FACTOR': args.prefetch_factor,
    'PIN_MEMORY': False,  # Force False to save memory
    'PIXEL_VALUE_MIN': int(os.getenv('PIXEL_VALUE_MIN', args.pixel_value_min)),
    'PIXEL_VALUE_MAX': int(os.getenv('PIXEL_VALUE_MAX', args.pixel_value_max)),
    'PIXEL_NORM_MIN': float(os.getenv('PIXEL_NORM_MIN', args.pixel_norm_min)),
    'PIXEL_NORM_MAX': float(os.getenv('PIXEL_NORM_MAX', args.pixel_norm_max)),
    'NUM_EPOCHS': args.max_epochs,
    'CACHE_DATASET': True if args.cache_dataset else False,
    'SMARTCACHE': args.smartcache,
    'SMARTCACHE_REPLACE_RATE': float(os.getenv('SMARTCACHE_REPLACE_RATE', '0.25')),
    'SMARTCACHE_INIT_WORKERS_FACTOR': float(os.getenv('SMARTCACHE_INIT_WORKERS_FACTOR', '0.5')),
    'SMARTCACHE_REPLACE_WORKERS_FACTOR': float(os.getenv('SMARTCACHE_REPLACE_WORKERS_FACTOR', '0.5')),
    'CACHE_RATE': args.cache_rate if hasattr(args, 'cache_rate') else float(os.getenv('CACHE_RATE', '1.0')),
    'PERSISTENT_DATASET': os.getenv('PERSISTENT_DATASET', 'True').lower() == 'true',
    'AVAILABLE_GPUs': torch.cuda.device_count(),
    'DEVICE_NO': int(os.getenv('DEVICE_NO', '0')),
    'VALIDATION_PHASE': bool(args.validation),
    'TRAINING_PHASE': bool(args.training),
    'TEST_PHASE': bool(args.testing or args.metric_testing),
    'USE_PRETRAINED': bool(args.use_pretrained),
    'WEIGHT_DIR': args.weights_dir,
    'ONE_HOT': args.one_hot,
    'IN_CHANNELS': args.in_channels,   
    'OUT_CHANNELS': args.out_channels,
    'PRECOMPUTE_SKELETONS': False,
    'SKEL_DIR': os.getenv('SKEL_DIR', None),  # Add SKEL_DIR, default to None if not in env
    'REGIONS_DIR': os.getenv('REGIONS_DIR', None), # Add REGIONS_DIR, default to None if not in env
    'MODEL_NAME': args.model_name,  # Add MODEL_NAME for anatomical cropping
    'CROP_BY_ANAT_STRUCTURES': bool(os.getenv('CROP_BY_ANAT_STRUCTURES', 'False').lower() == 'true'),
    'TRACHEA_PROB': float(os.getenv('TRACHEA_PROB', '0.15')),
    'BRANCH_PROB': float(os.getenv('BRANCH_PROB', '0.55')),
    'DISTAL_PROB': float(os.getenv('DISTAL_PROB', '0.30')),
    'CROP_DEBUG_SAVE': bool(os.getenv('CROP_DEBUG_SAVE', 'False').lower() == 'true'),
    'OUTPUT_PATH': os.getenv('OUTPUT_PATH', './outputs/'),
    'CROP_FAST_MODE': bool(os.getenv('CROP_FAST_MODE', 'True').lower() == 'true'),
    'MIN_BRANCH_CLUSTER_SIZE': int(os.getenv('MIN_BRANCH_CLUSTER_SIZE', '1')),
    'MIN_DISTAL_COMPONENT_SIZE': int(os.getenv('MIN_DISTAL_COMPONENT_SIZE', '1')),
    'MAX_CACHE_MEMORY_MB': int(os.getenv('MAX_CACHE_MEMORY_MB', '15000')),
    'FORCE_CACHE_NUM': int(os.getenv('FORCE_CACHE_NUM', '0')),
    'TEST_DATASET': args.test_dataset,
    'USE_PATCH_BANK_SAMPLING': bool(os.getenv('USE_PATCH_BANK_SAMPLING', 'True').lower() == 'true'),
    'PATCH_BANKS_DIR': os.getenv('PATCH_BANKS_DIR', './ATM22/npy_files/patch_banks/'),
    'SAMPLER_POS_RATIO': float(os.getenv('SAMPLER_POS_RATIO', '0.9')),
    'SAMPLER_TEMPERATURE': float(os.getenv('SAMPLER_TEMPERATURE', '0.8')),
    'SAMPLER_FG_THRESHOLD': float(os.getenv('SAMPLER_FG_THRESHOLD', '4.768e-05')),
    'SAMPLER_USE_LANDMARKS': bool(os.getenv('SAMPLER_USE_LANDMARKS', 'False').lower() == 'true'),
    'SAMPLER_QUOTA_BIFURCATION': float(os.getenv('SAMPLER_QUOTA_BIFURCATION', '0.0')),
    'SAMPLER_QUOTA_TRACHEA': float(os.getenv('SAMPLER_QUOTA_TRACHEA', '0.0')),
}

# Update params dict with SmartCache settings if enabled
if args.smartcache:
    params_dict['CACHE_DS'] = True
    params_dict['CACHE_RATE'] = args.cache_rate
    params_dict['SMARTCACHE_REPLACE_RATE'] = args.smartcache_replace_rate
    params_dict['SMARTCACHE_INIT_WORKERS_FACTOR'] = args.smartcache_init_workers_factor
    params_dict['SMARTCACHE_REPLACE_WORKERS_FACTOR'] = args.smartcache_replace_workers_factor

pin_memory = torch.cuda.is_available()
device = torch.device(f"cuda:{params_dict['DEVICE_NO']}" if torch.cuda.is_available() else "cpu")

exp_path = f"{args.loss_func}_" \
           f"{args.dataset}_" \
           f"{month}_{day}_{hour}_" \
           f"{args.model_name}_" \
           f"prtr{args.use_pretrained}_" \
           f"{args.optimizer}_" \
           f"{str(args.patch_size)}_" \
           f"b{str(args.batch_size)}_" \
           f"p{str(args.patience)}_" \
           f"{args.lr}_" \
           f"alpha{args.alpha}_" \
           f"r_d{args.r_d}" \
           f"r_l{args.r_l}" \
           f"{args.scheduler}"

if params_dict['VALIDATION_PHASE'] or params_dict['TEST_PHASE']:
    exp_path = os.getenv('OUTPUT_PATH', './outputs/') + str(args.exp_path)
    best_model_path = exp_path + ("/model/" if not args.awc_val else "/model_AWC/") + str(args.weights_path)
    output_path = os.getenv('OUTPUT_PATH', './outputs/') + args.exp_path
else:
    output_path = os.getenv('OUTPUT_PATH', './outputs/') + exp_path
    os.makedirs(output_path, exist_ok=True)

if params_dict['TRAINING_PHASE']:
    os.makedirs(output_path + "/model/", exist_ok=True)
    os.makedirs(output_path + "/outputs/", exist_ok=True)
    os.makedirs(output_path + "/val_outputs/", exist_ok=True)
    os.makedirs(output_path + "/off_val_outputs/", exist_ok=True)
    os.makedirs(output_path + "/off_test_outputs/", exist_ok=True)
    os.makedirs(output_path + "/visualization/", exist_ok=True)
    os.makedirs(output_path + "/AWC_STEP/", exist_ok=True)

if params_dict['TEST_PHASE'] or params_dict['VALIDATION_PHASE']:
    os.makedirs(output_path + "/preds/", exist_ok=True)
    if args.debug_inference:
        os.makedirs(output_path + "/debug_preds/", exist_ok=True)


data_path_for_datamodule = os.getenv('DATASET_PATH', './ATM22/npy_files/') # Directly use or default

if not os.path.isdir(data_path_for_datamodule):
    print(f"Warning: Dataset path {data_path_for_datamodule} determined from DATASET_PATH env or default construction does not exist.")
    # Fallback or error handling might be needed here if path is critical and not found

print(f"The output path is {output_path}")
print(f"The dataset root path for DataModule will be: {data_path_for_datamodule}")
print("parameters...")

for item in params_dict:
    print(f"{item} = {params_dict[item]}")

data_module = SegmentationDataModule(
    data_path=data_path_for_datamodule,
    batch_size=args.batch_size,
    dataset=args.dataset,
    params=params_dict,
    compute_dtype=torch.float16 if args.mixed else torch.float32,
    args=args
)

# Saved parameters to JSON for reproducibility
with open(output_path+"/params.json", 'w') as json_file:
    json.dump(params_dict, json_file, indent=2)
csv_file = output_path + "/results.csv"


model_params = {
    'MODEL_NAME': args.model_name,
    'NORM': args.norm,
    'SPATIAL_DIMS': args.spatial_dims,
    'IN_CHANNELS': args.in_channels,
    'OUT_CHANNELS': args.out_channels,
    'N_LAYERS': args.n_layers,
    'CHANNELS': args.channels,
    'STRIDES': args.strides,
    'NUM_RES_UNITS': args.num_res_units,
    'DROPOUT': args.dropout,
    'USE_PRETRAINED': args.use_pretrained,
    'WEIGHTS_PATH': args.weights_path,
    'TRAINING_PHASE': args.training  # Add training phase flag
}
# Saved the arguments to a JSON file
with open(output_path+"/model_params.json", 'w') as json_file:
    json.dump(model_params, json_file, indent=2)


class SegmentationNet(pl.LightningModule):
    def __init__(self, device: torch.device, data_module: pl.LightningDataModule):
        super().__init__()
        self.metric_hist = {}  # Initialize metric_hist
        self.csv_file = None   # Initialize csv_file
        self._device = device  # Use a different name to avoid conflict
        
        # Initialize best metrics
        self.best_val_dice = 0.0
        self.best_val_iou = 0.0
        self.best_val_epoch = 0 # Added to track epoch of best IoU
        
        # Set device type early for autocast compatibility
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model and model parameters
        self._model = get_model(params=model_params, device=self._device)
        
        # Set up mixed precision
        self.automatic_optimization = True
        
        # Initialize metrics
        self.train_dice_metric = get_metric(metric_name='DiceMetric', include_background=False, reduction="mean")
        self.val_dice_metric = get_metric(metric_name='DiceMetric', include_background=False, reduction="mean") 
        self.test_dice_metric = get_metric(metric_name='DiceMetric', include_background=False, reduction="mean")
        
        self.train_iou_metric = get_metric(metric_name='MeanIoU', include_background=False, reduction="mean")
        self.val_iou_metric = get_metric(metric_name='MeanIoU', include_background=False, reduction="mean")
        self.test_iou_metric = get_metric(metric_name='MeanIoU', include_background=False, reduction="mean")
        
        # Add metrics for test step
        self.meanDice_metric = get_metric(metric_name='DiceMetric', include_background=False, reduction="mean")
        self.meanIoU_metric = get_metric(metric_name='MeanIoU', include_background=False, reduction="mean")
        
        # Add airway-specific metrics
        self.airway_metrics = AirwayMetrics(include_background=True)
        
        # Toggle for heavy topology metrics to avoid long runtimes during validation
        self.topo_debug_mode = os.getenv('DEBUG_TOPOLOGY_METRICS', 'none').lower()
        self.enable_topology_metrics = self.topo_debug_mode not in ('none',)
        
        # Set up data module
        self.data_module = data_module
        
        # Initialize post-processing transforms
        self.post_pred_val = Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=float(os.getenv('PRED_THRESHOLD', args.pred_threshold)))
        ])
        # Use same post-processing for test and sc to ensure consistency
        self.post_pred_test = Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=float(os.getenv('PRED_THRESHOLD', args.pred_threshold)))
        ])
        #Here sc is for sliding window inference during validation
        self.post_pred_sc = Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=float(os.getenv('PRED_THRESHOLD', args.pred_threshold)))
        ])
        # The inverter will be initialized in `on_test_start`
        self.inverter = None
        
        # Initialize test step outputs list
        self.test_step_outputs = []
        
        # Initialize loss function (foreground only for binary segmentation)
        # Loss configuration with optional env overrides for composite losses
        self.loss_params = {}
        if args.loss_func == 'DiceCELoss':
            try:
                self.loss_params["lambda_dice"] = float(os.getenv('DICECE_LAMBDA_DICE', '1.0'))
                self.loss_params["lambda_ce"] = float(os.getenv('DICECE_LAMBDA_CE', '0.5'))
            except Exception:
                self.loss_params = {"lambda_dice": 1.0, "lambda_ce": 0.5}
        self.loss_function = get_loss(args.loss_func, params=self.loss_params, include_background=False)
        
        
        # Memory optimization
        if args.gradient_checkpointing:
            print("Enabling gradient checkpointing for memory optimization...")
            self._apply_checkpointing(self._model)
            
        # Initialize counters for debugging
        self.batch_count = 0

    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for m in self._model.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Applied Kaiming initialization to model weights")

    def _apply_checkpointing(self, module):
        """Helper to apply gradient checkpointing to submodules"""
        if len(list(module.children())) > 0:
            # Only apply to modules with parameters 
            if any(p.requires_grad for p in module.parameters()):
                # Check if it's a good candidate for checkpointing
                if isinstance(module, (nn.Sequential, nn.ModuleList)) or "Block" in module.__class__.__name__:
                    # Skip embedding layers and small layers
                    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    if param_count > 1000:  # Only checkpoint reasonably sized modules
                        module._original_forward = module.forward
                        module.forward = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                            module._original_forward, *args, **kwargs, use_reentrant=False
                        )
                        print(f"Applied checkpointing to {module.__class__.__name__}")

    def forward(self, x):
        if not self.training:
            # For validation/testing, typically no deep supervision needed for inference part of LightningModule
            with torch.no_grad():
                with amp.autocast(enabled=self.trainer.precision == '16-mixed', dtype=torch.float16, device_type=self.device_type):
                    val_output = self._model(x)
                    if self.batch_count < 5: # Only check first few batches
                        is_on_gpu = verify_tensor_on_gpu(val_output, "Validation Output tensor")
                        if not is_on_gpu:
                            print(f"Warning: Validation output tensor not on GPU for input shape {x.shape}")
                    return val_output

        # Training mode forward pass
        if self.batch_count % 100 == 0:  # Reduced frequency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if not x.is_contiguous():
            x = x.contiguous()
            
        with torch.amp.autocast(enabled=self.trainer.precision == '16-mixed', dtype=torch.float16, device_type=self.device_type):
            model_outputs = self._model(x)

            # The new SimpleResUNet and BasicUNet both return single tensors, no deep supervision
            if isinstance(model_outputs, torch.Tensor):
                if self.batch_count < 5 and not verify_tensor_on_gpu(model_outputs, "Model output"):
                    raise RuntimeError("Model output is not on GPU!")
                return model_outputs
            else:
                # Handle any unexpected output format
                if self.batch_count < 5:
                    print(f"[WARNING] Model returned unexpected output type: {type(model_outputs)}")
                if isinstance(model_outputs, (list, tuple)) and len(model_outputs) > 0:
                    # If it's a tuple/list, take the first element
                    return model_outputs[0]
                else:
                    raise RuntimeError(f"Model forward returned unexpected type: {type(model_outputs)}")

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        if args.metric_testing:
            # For metric testing, use the dataloader with ground-truth labels (from the validation split of the training set)
            print("Metric testing enabled: using validation dataloader with labels.")
            return self.data_module.aff_training_val_dataloader()
        else:
            # For inference, use the dataloader without labels (official validation set)
            print("Inference mode: using official test dataloader (no labels).")
            return self.data_module.off_test_dataloader()

    def off_val_dataloader(self):
        return self.data_module.off_val_dataloader()

    def off_test_dataloader(self):
        # In testing mode, we may not have labels, so handle this case
        if args.metric_testing:
            # For metric testing, use the dataloader with labels
            return self.data_module.test_dataloader()
        else:
            # For inference, use the dataloader without labels
            return self.data_module.off_test_dataloader()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        """
        # Define optimizer with gradient clipping
        opt = torch.optim.AdamW(self.parameters(), lr=float(args.lr), weight_decay=1e-4)
        # Define scheduler based on type
        total_steps = self.trainer.estimated_stepping_batches
        if args.scheduler == "CosineAnnealing":
            try:
                scheduler = get_scheduler(
                    optimizer=opt,
                    type=args.scheduler, 
                    T_max=args.max_epochs
                )
                print("Using CosineAnnealing scheduler with T_max =", args.max_epochs)
                return {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_iou",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            except Exception as e:
                print(f"Error setting up CosineAnnealing scheduler: {e}")
                print("Falling back to optimizer without scheduler")
                return opt
        elif args.scheduler == "WarmupCosineSchedule":
            # Warm restarts per PyTorch docs
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
            scheduler = WarmupCosineSchedule(
                optimizer=opt,
                warmup_steps=int(0.05 * total_steps),
                t_total=total_steps,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_iou",
                    "interval": "step",   # step every optimizer step (recommended)
                    "frequency": 1,
                },
            }       
        elif args.scheduler == "ReduceLROnPlateau":
            scheduler = get_scheduler( # Assuming get_scheduler correctly sets up ReduceLROnPlateau
                optimizer=opt,
                type=args.scheduler,
                patience=args.patience, # This patience is in terms of epochs where val_iou doesn't improve
            )
            print("Using ReduceLROnPlateau scheduler")
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_iou",   # Correct metric
                    "interval": "epoch",  # Check at the end of each epoch
                    "frequency": 1,       # Check every 1 interval (i.e., every epoch)
                },
            }
        elif args.scheduler == "OneCycleLR":
            # Use OneCycleLR with warmup for better convergence
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr, total_steps=total_steps,
                pct_start=0.1, div_factor=10, final_div_factor=100
            )
            print("Using OneCycleLR scheduler with warmup")
            return {"optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_iou", "frequency": 1}}
        else:
            print("No scheduler used")
            return opt
            
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step(closure=optimizer_closure)

    def _clear_memory(self):
        """Helper method to centralize memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_start(self):
        print("=== TRAINING START SEQUENCE ===")
        sequence_start = time.time()
        
        self.training_start_time = time.time()
        print(f"[{time.time() - sequence_start:.1f}s] Initializing training start sequence...")
        
        #Force to ensure model is in GPU
        print(f"[{time.time() - sequence_start:.1f}s] Moving model to GPU...")
        self._model = force_model_to_gpu(self._model, self._device)
        diagnose_model_placement(self._model)
        monitor_gpu_usage()
        
        if self.trainer.is_global_zero:
            global csv_file
            self.csv_file = csv_file
        
        # Start SmartCacheDataset if using caching
        if hasattr(self.data_module, 'train_ds') and hasattr(self.data_module.train_ds, 'start'):
            print(f"[{time.time() - sequence_start:.1f}s] Starting SmartCacheDataset background replacement thread...")
            self.data_module.train_ds.start()
        if hasattr(self.data_module, 'val_ds') and hasattr(self.data_module.val_ds, 'start'):
            print(f"[{time.time() - sequence_start:.1f}s] Starting validation SmartCacheDataset background replacement thread...")
            self.data_module.val_ds.start()
            
        print(f"[{time.time() - sequence_start:.1f}s] Training start sequence completed. Ready for sanity check...")
        print("=== END TRAINING START SEQUENCE ===")

    def on_train_epoch_start(self):
        """Update SmartCache before each epoch."""
        if hasattr(self.data_module, 'train_ds') and hasattr(self.data_module.train_ds, 'update_cache'):
            print(f"Updating SmartCache for epoch {self.current_epoch}...")
            self.data_module.train_ds.update_cache()
        if hasattr(self.data_module, 'val_ds') and hasattr(self.data_module.val_ds, 'update_cache'):
            self.data_module.val_ds.update_cache()

    def on_train_end(self):
        # Shutdown SmartCacheDataset if using caching
        if hasattr(self.data_module, 'train_ds') and hasattr(self.data_module.train_ds, 'shutdown'):
            print("Shutting down SmartCacheDataset background replacement thread...")
            self.data_module.train_ds.shutdown()
        if hasattr(self.data_module, 'val_ds') and hasattr(self.data_module.val_ds, 'shutdown'):
            print("Shutting down validation SmartCacheDataset background replacement thread...")
            self.data_module.val_ds.shutdown()
            
        if self.trainer.is_global_zero:
            # Save metric history as JSON
            metrics_filename = os.path.join(self.trainer.log_dir, "metrics_history.json")
            try:
                # Convert tensors to lists for JSON serialization if they exist in metric_hist
                serializable_metric_hist = {}
                for key, value_list in self.metric_hist.items():
                    serializable_metric_hist[key] = [v.item() if isinstance(v, torch.Tensor) else v for v in value_list]
                
                with open(metrics_filename, 'w') as f:
                    json.dump(serializable_metric_hist, f, indent=4)
                print(f"Metrics history saved to {metrics_filename}")
            except Exception as e:
                print(f"Could not save metrics history: {e}")

            best_model_cb = None
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    best_model_cb = cb
                    break
            
            epoch_num_str = "N/A"
            if best_model_cb and hasattr(best_model_cb, 'best_model_path') and best_model_cb.best_model_path:
                try:
                    filename = os.path.basename(best_model_cb.best_model_path)
                    # Try to find 'epoch=XX' first
                    epoch_match = re.search(r"epoch=(\d+)", filename)
                    if epoch_match:
                        epoch_num_str = epoch_match.group(1)
                    else:
                        # This regex looks for a number after 'epoch_' or '-epoch_'
                        # and also considers if it's followed by a known metric like '-iou' or '-dice'
                        # Example: model-epoch_44-iou_0.2112.ckpt, model_epoch_44_dice_0.000.ckpt
                        # More general: Look for 'epoch_NUM' or '-NUM-' where NUM is likely an epoch
                        specific_patterns = [
                            r"epoch=(\d+)[-_.]",          
                            r"epoch_(\d+)[-_.]",         
                            r"-epoch_(\d+)[-_.]",        
                            r"_epoch_(\d+)[-_.]"         
                        ]
                        for pat in specific_patterns:
                            match = re.search(pat, filename)
                            if match:
                                epoch_num_str = match.group(1)
                                break
                        if epoch_num_str == "N/A": # If still not found, try more general number extraction
                            # Look for numbers that are likely epochs (e.g., not part of a float like 0.2112)
                            # This is a bit more heuristic
                            potential_epochs = re.findall(r"[_=-](\d+)[-_.]", filename) # Numbers surrounded by delimiters
                            if not potential_epochs:
                                potential_epochs = re.findall(r"[_=-](\d+).ckpt", filename) # Number before .ckpt
                            
                            if potential_epochs:
                                # Filter out numbers that look like metric values (e.g., have many digits or are small)
                                valid_epochs = [p for p in potential_epochs if len(p) < 5 and not (len(p) > 1 and p.startswith('0'))] 
                                if valid_epochs:
                                     epoch_num_str = valid_epochs[-1] # Take the last plausible one
                except Exception as e:
                    print(f"Error parsing epoch from checkpoint filename {filename}: {e}")
                    epoch_num_str = "N/A (parsing error)"

            print(f"Training completed.")
            # Ensure self.best_val_iou and self.best_val_dice are floats for printing
            if isinstance(self.best_val_iou, torch.Tensor):
                best_iou_to_print = float(self.best_val_iou.item())
            else:
                best_iou_to_print = float(self.best_val_iou)
            
            if isinstance(self.best_val_dice, torch.Tensor):
                best_dice_to_print = float(self.best_val_dice.item())
            else:
                best_dice_to_print = float(self.best_val_dice)

            print(f"Best validation IoU: {best_iou_to_print:.4f} (achieved around epoch {epoch_num_str}).")
            print(f"Validation Dice score at that point: {best_dice_to_print:.4f}.")
            
            # Close CSV file if open
            if hasattr(self, 'csv_logger_file') and self.csv_logger_file is not None: # Check if it's not None
                if hasattr(self.csv_logger_file, 'close') and callable(self.csv_logger_file.close):
                    try:
                        self.csv_logger_file.close()
                        print("Epoch metrics CSV file closed.")
                    except Exception as e:
                        print(f"Error closing epoch metrics CSV file: {e}")
                else:
                    # This case indicates self.csv_logger_file might be a path string or something else
                    print(f"Warning: self.csv_logger_file (type: {type(self.csv_logger_file)}) does not have a callable 'close' method.")
            
            if hasattr(self, 'csv_file_handle') and self.csv_file_handle is not None: 
                try:
                    if hasattr(self.csv_file_handle, 'close') and callable(self.csv_file_handle.close):
                        self.csv_file_handle.close() 
                        print("Epoch metrics CSV file (csv_file_handle) closed.")
                        self.csv_file_handle = None 
                    else:
                        print(f"Warning: self.csv_file_handle (type: {type(self.csv_file_handle)}) does not have a callable 'close' method.")
                except Exception as e:
                    print(f"Error closing CSV file handle in on_train_end: {e}")

        self._clear_memory() # Calls gc.collect() and cuda.empty_cache()
        monitor_gpu_usage(f"End of training ({self.current_epoch})")

    def training_step(self, batch, batch_idx):
        self.batch_count += 1
        if batch_idx % 100 == 0:
            self._clear_memory()
        loss = None
        try:
            inputs = batch["image"].to(self._device, non_blocking=True)
            labels = batch["label"].to(self._device, non_blocking=True)
            if batch_idx == 0:
                if not verify_tensor_on_gpu(inputs, "Input batch"):
                    print("Input batch was on CPU! Moving to GPU...")
                    inputs = inputs.to(self._device)
                if not verify_tensor_on_gpu(labels, "Label batch"):
                    print("Label batch was on CPU! Moving to GPU...")
                    labels = labels.to(self._device)
            
            if batch_idx == 0 or batch_idx % 50 == 0:
                print(f"Training - Input Shape: {inputs.shape}, Label Shape: {labels.shape}")
            label_sum = torch.sum(labels > 0).item()
            bg_only_batch = (label_sum == 0)
            if bg_only_batch and (batch_idx == 0 or batch_idx % 50 == 0):
                print(f"INFO: Batch {batch_idx} has no positive labels. Using background-only BCE loss.")               

            if args.loss_func in ['GUL', 'BEL', 'CBEL']:
                lungs = batch["lung"].to(self._device, non_blocking=True) if "lung" in batch else None
                weights = batch["weight"].to(self._device, non_blocking=True) if "weight" in batch else None


            with amp.autocast(enabled=args.mixed, dtype=torch.float16, device_type=self.device_type):
                if self._model.__class__.__name__ == 'SimpleResUNet' and self.training:
                    logits = self.forward(inputs)
                else:
                    logits = self.forward(inputs)
            
                if batch_idx == 0 or batch_idx % 50 == 0:
                    print(f"Training - Model Output Shape: {logits.shape}")
            
                labels = labels.to(dtype=logits.dtype)
            # Calculate loss
            if bg_only_batch:
                # Background-only supervision: penalize any false positives robustly
                bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                zeros = torch.zeros_like(labels, dtype=logits.dtype)
                loss = bce_loss_fn(logits, zeros)
            else:
                # Calculate loss based on the loss function type
                if args.loss_func == 'GUL' or args.loss_func == 'BEL':
                    if weights is None:
                        raise ValueError(f"Loss function {args.loss_func} requires 'weight' in batch but it was not found.")
                    # For custom loss functions that take weights, pass them as keyword args
                    try:
                        loss = self.loss_function(logits.float(), labels.float(), weight_map=weights)
                    except TypeError:
                        # Fallback to standard 2-argument call if weight parameter isn't supported
                        loss = self.loss_function(logits.float(), labels.float())
                elif args.loss_func == 'CBEL':
                    if weights is None:
                        raise ValueError(f"Loss function {args.loss_func} requires 'weight' in batch but it was not found.")
                    if lungs is None:
                        raise ValueError(f"Loss function {args.loss_func} requires 'lung' in batch but it was not found.")
                    # For custom loss functions that take weights and lungs, pass them as keyword args
                    try:
                        loss = self.loss_function(logits.float(), labels.float(), weight_map=weights, lung_mask=lungs)
                    except TypeError:
                        # Fallback to standard 2-argument call if parameters aren't supported
                        loss = self.loss_function(logits.float(), labels.float())
                else:
                    loss = self.loss_function(logits.float(), labels.float())
        
            # Less frequent memory usage reports to reduce overhead
            if batch_idx % 10 == 0:
                memory_usage = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[Batch {batch_idx}] Memory: {memory_usage:.2f} GB, Reserved: {reserved:.2f} GB")
            
            current_loss_val = loss.item()
            self.log("train_loss", current_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False, batch_size=args.batch_size)
            
            upper_warn = 0.999
            if args.loss_func.lower().startswith("dicece"):
                # CE starts ~0.55–0.70; with lambda_ce (default 0.5), upper bound ≈ 1 + 0.5*0.70 = 1.35
                upper_warn = 1.0 + self.loss_params.get("lambda_ce", 0.5) * 0.70
            if current_loss_val < 1e-4 or current_loss_val > upper_warn:
                print("WARNING: Unusual loss value detected:", current_loss_val)
            # Explicitly delete tensors to help with memory management
            del inputs, labels, logits # Ensure logits is deleted
            if args.loss_func in ['GUL', 'BEL', 'CBEL']:
                del lungs, weights
                return loss
        except Exception as e:
            print(f"Error during training step: {e}")
            self._clear_memory()
            raise e
        return loss

    def validation_step(self, batch, batch_idx):
        # Clear memory before validation step
        self._clear_memory()
        
        val_inputs = batch["image"].to(self._device, non_blocking=True)
        val_labels = batch["label"].to(self._device, non_blocking=True)
        
        # Load additional data if needed by GUL/BEL/CBEL loss functions
        val_lungs = None
        val_weights = None
        if args.loss_func in ['GUL', 'BEL', 'CBEL']:
            val_lungs = batch["lung"].to(self._device, non_blocking=True) if "lung" in batch else None
            val_weights = batch["weight"].to(self._device, non_blocking=True) if "weight" in batch else None
        
        # Get filename for debugging - ALWAYS get it for better tracking
        val_filename = "Unknown"
        if 'image_meta_dict' in batch and 'filename_or_obj' in batch['image_meta_dict']:
            try:
                val_filename = batch['image_meta_dict']['filename_or_obj'][0] 
                if isinstance(val_filename, (list, tuple)):
                    val_filename = str(val_filename[0])
                else:
                    val_filename = str(val_filename)
                # Extract just the filename from full path
                val_filename = os.path.basename(val_filename)
            except Exception:
                pass
        
        # More efficient check for empty batches
        label_sum_on_gpu = torch.sum(val_labels > 0).item()
        if label_sum_on_gpu == 0:
            self.empty_batches += 1
            print(f"Warning: Validation batch {batch_idx} (Filename: {val_filename}) has no positive labels. Skipping metrics logging.")
            # Immediate cleanup including GUL/BEL/CBEL data
            del val_inputs, val_labels
            if val_lungs is not None:
                del val_lungs
            if val_weights is not None:
                del val_weights
            torch.cuda.empty_cache()

            # Return None to indicate this batch should be skipped from aggregation
            return None

        if batch_idx < 3:
            print(f"Validation Image Stats - Min: {val_inputs.min():.4f}, Max: {val_inputs.max():.4f}")
            print(f"Validation Label Stats - Positive Pixels: {label_sum_on_gpu}")

        with torch.no_grad():
            torch.cuda.empty_cache()
            with amp.autocast(enabled=args.mixed, dtype=torch.float16, device_type=self.device_type):
                raw_val_roi_size = args.test_patch_size if args.test_patch_size is not None else args.patch_size
                val_roi_size = (raw_val_roi_size,) * 3 if isinstance(raw_val_roi_size, int) else raw_val_roi_size
                sw_batch_size = 1
                ovlp = 0.5
                banks_dir = os.getenv("PATCH_BANKS_DIR", "./ATM22/npy_files/patch_banks")
                use_weighted = bool(int(os.getenv("USE_WEIGHTED_AGGREGATOR", "1")))
                mode = os.getenv("CONF_SOURCE", "audit").lower()  # "audit" | "landmarks"

                if use_weighted:
                    # case id from meta (consistent with the bank sampler)
                    case_id = extract_case_id_from_meta(batch.get("image_meta_dict", {}))
                    patch_list, meta = load_patch_bank_for_case(case_id, banks_dir)
                    if patch_list:
                        # build provider
                        if mode == "landmarks":
                            # processed spacing if available in meta_dict (optional)
                            spacing = None
                            try:
                                # MONAI MetaTensor sometimes carries pixdim via affine; you can pass None to treat σ as voxels
                                spacing = None
                            except Exception:
                                spacing = None
                            per_w = make_landmark_weight_provider_from_bank(
                                patch_list=patch_list,
                                vol_shape_zyx=tuple(int(v) for v in val_inputs.shape[-3:]),
                                roi_zyx=tuple(int(v) for v in val_roi_size),
                                spacing_zyx_mm=spacing,
                                sigma_mm=float(os.getenv("CONF_SIGMA_MM", "30.0")),
                            )
                        else:
                            per_w = make_audit_weight_provider(
                                patch_list=patch_list, roi_zyx=tuple(int(v) for v in val_roi_size)
                            )

                        outputs = weighted_sliding_window_inference(
                            inputs=val_inputs,
                            predictor=self.forward,
                            roi_size=tuple(int(v) for v in val_roi_size),
                            overlap=ovlp,
                            per_window_weight=per_w,
                            device=self._device,
                            gaussian_blend=True,
                        )
                    else:
                        # Fallback to standard SWI if bank missing
                        outputs = sliding_window_inference(
                            inputs=val_inputs,
                            roi_size=val_roi_size,
                            sw_batch_size=1,
                            predictor=self.forward,
                            overlap=ovlp,
                            mode="gaussian",
                            cval=0
                        )
                else:
                    outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=val_roi_size,
                        sw_batch_size=1,
                        predictor=self.forward,
                        overlap=ovlp,
                        mode="gaussian",
                        cval=0
                    )
                    
            if batch_idx % 20 == 0:
                print(f"Val batch {batch_idx}: Input {val_inputs.shape}, Output {outputs.shape}")
            torch.cuda.empty_cache()

        # Ensure labels match outputs dtype to avoid unnecessary upcasts
        val_labels = val_labels.to(dtype=outputs.dtype)
        # Calculate loss using raw outputs and original labels
        loss_val = self.loss_function(outputs, val_labels)
        self.log('val_loss_step', loss_val) # Log the step loss

        # Post-process outputs for metric calculation (sigmoid + threshold)
        val_outputs_metric = self.post_pred_sc({"pred": outputs})["pred"] # Applies sigmoid and AsDiscrete
        val_labels_metric = val_labels.byte() # Ensure labels are byte for MONAI metrics

        # Update metric objects
        self.val_dice_metric(y_pred=val_outputs_metric, y=val_labels.byte())
        self.val_iou_metric(y_pred=val_outputs_metric, y=val_labels.byte())
        
        # Update airway-specific metrics with timing (optional, can be very heavy)
        if self.enable_topology_metrics:
            if batch_idx < 3:  # Only show timing for first few batches to avoid spam
                metrics_start = time.time()
                print(f"[Batch {batch_idx}] Starting topology metrics calculation...")
            
            val_lungs_metric = val_lungs.byte() if val_lungs is not None else None
            # Pass filename for better debugging of problematic samples
            self.airway_metrics.update(pred=val_outputs_metric, target=val_labels.byte(), lungs=val_lungs_metric, filenames=[val_filename])
            
            if batch_idx < 3:
                metrics_time = time.time() - metrics_start
                print(f"[Batch {batch_idx}] Topology metrics completed in {metrics_time:.2f}s")
        else:
            if batch_idx == 0:
                print("Topology metrics disabled via DEBUG_TOPOLOGY_METRICS. Skipping heavy computations in validation.")

        # Log step loss for epoch aggregation by PyTorch Lightning
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=val_inputs.size(0))
        
        self.valid_batches +=1 # Increment valid batches counter

        del val_inputs, val_labels, outputs, val_outputs_metric, val_labels_metric
        # ... (GUL/BEL/CBEL data cleanup if any) ...
        if batch_idx % 1 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        return loss_val

    def on_validation_epoch_start(self):
        """Clear memory before validation epoch starts."""
        print(f"=== VALIDATION EPOCH {self.current_epoch} START ===")
        val_start_time = time.time()
        
        # Initialize counters for tracking validation batches
        self.valid_batches = 0
        self.empty_batches = 0
        self.validation_start_time = val_start_time
        self.problematic_samples = []  # Track samples with IoU=0
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        print(f"[{time.time() - val_start_time:.1f}s] Validation epoch {self.current_epoch} initialized")

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        
        total_batches = self.valid_batches + self.empty_batches
        if total_batches > 0:
            empty_batch_percentage = (self.empty_batches / total_batches) * 100
            print(f"Validation batch stats: {self.valid_batches} valid (processed), {self.empty_batches} empty ({empty_batch_percentage:.1f}% empty and skipped)")
            
            if self.empty_batches > 0 and self.empty_batches == total_batches:
                print("WARNING: All validation batches were empty. Metrics will be zero. Check validation data and preprocessing.")
                self.log("val_dice", 0.0, prog_bar=True, logger=True)
                self.log("val_iou", 0.0, prog_bar=True, logger=True)
                val_loss_epoch = self.trainer.logged_metrics.get('val_loss', torch.tensor(float('nan'))).item()
                if math.isnan(val_loss_epoch): 
                    val_loss_epoch = 1.0 
                    self.log('val_loss', val_loss_epoch, on_epoch=True, prog_bar=True, logger=True)

                self.val_dice_metric.reset()
                self.val_iou_metric.reset()
                return

        # Aggregate metrics only if there were valid batches processed by the metric objects
        val_dice_epoch = self.val_dice_metric.aggregate().item() if self.valid_batches > 0 else 0.0
        val_iou_epoch = self.val_iou_metric.aggregate().item() if self.valid_batches > 0 else 0.0
        val_loss_epoch = self.trainer.logged_metrics.get('val_loss', torch.tensor(0.0)).item()

        # Log traditional metrics
        self.log("val_dice", val_dice_epoch, prog_bar=True, logger=True)
        self.log("val_iou", val_iou_epoch, prog_bar=True, logger=True)

        # Calculate and log the "hard" dice loss
        val_dice_loss_epoch = 1.0 - val_dice_epoch
        self.log("val_dice_loss", val_dice_loss_epoch, prog_bar=True, logger=True)
        
        # Compute and log airway-specific metrics (optional)
        if self.enable_topology_metrics:
            airway_metrics = self.airway_metrics.compute()
            self.log("val_trachea_dice", airway_metrics["trachea_dice"].item(), prog_bar=False, logger=True)
            self.log("val_dlr", airway_metrics["dlr"].item(), prog_bar=False, logger=True)  # Detected Length Ratio
            self.log("val_dbr", airway_metrics["dbr"].item(), prog_bar=False, logger=True)  # Detected Branch Ratio
            self.log("val_precision", airway_metrics["precision"].item(), prog_bar=False, logger=True)
            self.log("val_leakage", airway_metrics["leakage"].item(), prog_bar=False, logger=True)
            self.log("val_amr", airway_metrics["amr"].item(), prog_bar=False, logger=True)  # Airway Missing Ratio
            self.airway_metrics.reset()
        else:
            pass
        
        # Reset other metrics
        self.val_dice_metric.reset()
        self.val_iou_metric.reset()
        
        # Track best IoU, Dice, and epoch
        if hasattr(self, 'best_val_iou'): # Check if initialized
            if val_iou_epoch > self.best_val_iou:
                self.best_val_iou = val_iou_epoch
                self.best_val_dice = val_dice_epoch # Store the corresponding Dice score
                self.best_val_epoch = self.current_epoch
                print(f"New Best Val IoU: {self.best_val_iou:.4f} (Dice: {self.best_val_dice:.4f}) at epoch {self.best_val_epoch}")
        else: # Initialize if it's the first validation epoch end
            self.best_val_iou = val_iou_epoch
            self.best_val_dice = val_dice_epoch
            self.best_val_epoch = self.current_epoch
            print(f"Initial Best Val IoU: {self.best_val_iou:.4f} (Dice: {self.best_val_dice:.4f}) at epoch {self.best_val_epoch}")

        print(f"Epoch {self.current_epoch}: Val Loss: {val_loss_epoch:.4f}, Val Dice Epoch: {val_dice_epoch:.4f}, Val IoU Epoch: {val_iou_epoch:.4f}")
        
        # Report problematic samples if any
        if hasattr(self, 'problematic_samples') and self.problematic_samples:
            print(f"\nWARNING: Found {len(self.problematic_samples)} problematic samples with very low IoU:")
            for sample_info in self.problematic_samples[:5]:  # Show first 5
                print(f"  - {sample_info}")
            if len(self.problematic_samples) > 5:
                print(f"  ... and {len(self.problematic_samples) - 5} more")
            problematic_file = os.path.join(output_path, f"problematic_samples_epoch_{self.current_epoch}.txt")
            with open(problematic_file, 'w') as f:
                for sample_info in self.problematic_samples:
                    f.write(f"{sample_info}\n")
            print(f"  Full list saved to: {problematic_file}")
        
        torch.cuda.empty_cache()
        
    def on_test_start(self):
        self.test_step_outputs = []
        try:
            import torch
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    def test_step(self, batch, batch_idx):
        tst_inputs = batch["image"].to(self._device)
        thr = float(os.getenv("PRED_THRESHOLD", getattr(args, "pred_threshold", 0.5))) #added for manual threshold control
        with torch.no_grad():
            # Determine ROI size for inference
            raw_test_roi_size = args.test_patch_size if args.test_patch_size is not None else args.patch_size
            test_roi_size = (raw_test_roi_size,) * 3 if isinstance(raw_test_roi_size, int) else raw_test_roi_size
            print(f"Using test ROI size: {test_roi_size}")
            ovlp = 0.5 # Increased overlap for smoother predictions
            banks_dir = os.getenv("PATCH_BANKS_DIR", "./ATM22/npy_files/patch_banks")
            use_weighted = bool(int(os.getenv("USE_WEIGHTED_AGGREGATOR", "1")))
            mode = os.getenv("CONF_SOURCE", "audit").lower()  # "audit" | "landmarks"
            if use_weighted:
                # case id from meta (consistent with the bank sampler)
                case_id = extract_case_id_from_meta(batch.get("image_meta_dict", {}))
                patch_list, meta = load_patch_bank_for_case(case_id, banks_dir)

                if patch_list:
                    # build provider
                    if mode == "landmarks":
                        # processed spacing if available in meta_dict (optional)
                        spacing = None
                        try:
                            # MONAI MetaTensor sometimes carries pixdim via affine; you can pass None to treat σ as voxels
                            spacing = None
                        except Exception:
                            spacing = None
                        per_w = make_landmark_weight_provider_from_bank(
                            patch_list=patch_list,
                            vol_shape_zyx=tuple(int(v) for v in tst_inputs.shape[-3:]),
                            roi_zyx=tuple(int(v) for v in test_roi_size),
                            spacing_zyx_mm=spacing,
                            sigma_mm=float(os.getenv("CONF_SIGMA_MM", "30.0")),
                        )
                    else:
                        per_w = make_audit_weight_provider(
                            patch_list=patch_list, roi_zyx=tuple(int(v) for v in test_roi_size)
                        )

                    logits = weighted_sliding_window_inference(
                        inputs=tst_inputs,
                        predictor=self.forward,
                        roi_size=tuple(int(v) for v in test_roi_size),
                        overlap=ovlp,
                        per_window_weight=per_w,
                        device=self._device,
                        gaussian_blend=True,
                    )
                else:
                    # Fallback to standard SWI if bank missing
                    logits = sliding_window_inference(
                        inputs=tst_inputs,
                        roi_size=test_roi_size,
                        sw_batch_size=1,
                        predictor=self.forward,
                        overlap=ovlp,
                        mode="gaussian",
                        cval=0
                    )
            else:
                logits = sliding_window_inference(
                    inputs=tst_inputs,
                    roi_size=test_roi_size,
                    sw_batch_size=1,
                    predictor=self.forward,
                    overlap=ovlp,
                    mode="gaussian",
                    cval=0
                )

        # Decollate batch to process each item individually
        decollated_batch = decollate_batch(batch)
        decollated_logits = decollate_batch(logits)

        for i, (item, logit_item) in enumerate(zip(decollated_batch, decollated_logits)):
            # Combine original batch data with its corresponding prediction logit
            pred_mt = MetaTensor(logit_item)
            try:
                if isinstance(item.get("image"), MetaTensor) and hasattr(item["image"], "meta"):
                    pred_mt.meta = item["image"].meta.copy()
                    if hasattr(item["image"], "affine") and item["image"].affine is not None:
                        pred_mt.affine = item["image"].affine.clone() if hasattr(item["image"].affine, "clone") else item["image"].affine
            except Exception:
                pass
            item["pred"] = pred_mt
            # Debug:Print shapes before inversion
            print(f"\n[DEBUG] Processing item {i}:")
            print(f"  Prediction shape before inversion: {item['pred'].shape}")
            print(f"  Image shape: {item['image'].shape}")
            if 'lung' in item and item['lung'] is not None:
                print(f"  Lung shape: {item['lung'].shape}")
            
            # Preserve the lung mask at the processed resolution (before inversion)
            # This is needed for post-processing to remove predictions outside lungs
            lung_processed = None
            if "lung" in item and item["lung"] is not None:
                lung_processed = item["lung"].clone()  # Keep a copy at processed resolution
            
            # Save raw logits and thresholded prediction before inversion
            if getattr(args, "debug_inference", False):
                debug_dir = os.path.join(output_path, "debug_preds")
                os.makedirs(debug_dir, exist_ok=True)

                filename = os.path.basename(item.get("image_meta_dict", {}).get("filename_or_obj", f"unknown_{batch_idx}_{i}"))
                base_name = filename.replace(".npy", "").replace(".nii.gz", "")

                import nibabel as nib
                logits_np = item["pred"].cpu().numpy().squeeze()
                affine = item["pred"].affine.cpu().numpy() if hasattr(item["pred"], "affine") else np.eye(4)
                nib.save(nib.Nifti1Image(logits_np, affine), os.path.join(debug_dir, f"{base_name}_1_raw_logits.nii.gz"))

                sigmoid_np = torch.sigmoid(item["pred"]).cpu().numpy().squeeze()
                nib.save(nib.Nifti1Image(sigmoid_np, affine), os.path.join(debug_dir, f"{base_name}_2_sigmoid.nii.gz"))
                binary_np = (sigmoid_np > thr).astype(np.uint8)
                nib.save(nib.Nifti1Image(binary_np, affine), os.path.join(debug_dir, f"{base_name}_3_binary_pre_invert.nii.gz"))
                print("  DEBUG: Saved pre-inversion files to debug_preds/")
            
            
            # --- POST-PROCESSING PIPELINE ---
            # 1) Activation + threshold in processed space
            preinvert_pipeline = Compose([
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=float(os.getenv('PRED_THRESHOLD', args.pred_threshold))),
            ])
            item = preinvert_pipeline(item)

            if getattr(args, "metric_testing", False):
                label_np = item["label"].cpu().numpy().squeeze().astype(np.uint8)
                pred_np = item["pred"].cpu().numpy().squeeze().astype(np.uint8)

                if "lung" in item and item["lung"] is not None:
                    lungs_np = (item["lung"].cpu().numpy().squeeze() > 0).astype(np.uint8)
                else:
                    lungs_np = np.ones_like(label_np, dtype=np.uint8)

                iou, dlr, dbr, precision, leakage, amr, *_tmp, largest_cc = evaluation_branch_metrics(
                    fid=item.get("image_meta_dict", {}).get("filename_or_obj", "unknown"),
                    label=label_np,
                    pred=pred_np,
                    lungs=lungs_np,
                    refine=False,
                )
                intersection_full = float(((pred_np > 0) & (label_np > 0)).sum())
                denom_full = float((pred_np > 0).sum() + (label_np > 0).sum())
                dice_full = (2.0 * intersection_full / denom_full) if denom_full > 0 else 0.0
                union_full = float((pred_np > 0).sum() + (label_np > 0).sum() - ((pred_np > 0) & (label_np > 0)).sum())
                iou_full = (intersection_full / union_full) if union_full > 0 else 0.0

                # LCC dice (matching the LCC IoU that eval returns)
                dice_lcc = 0.0
                if largest_cc is not None:
                    lcc = (largest_cc > 0).astype(np.uint8)
                    gt = (label_np > 0).astype(np.uint8)
                    inter_lcc = float((lcc & gt).sum())
                    denom_lcc = float(lcc.sum() + gt.sum())
                    dice_lcc = (2.0 * inter_lcc / denom_lcc) if denom_lcc > 0 else 0.0
                metrics = {
                    "iou_full": iou_full,
                    "dice_full": dice_full,
                    "iou_lcc": float(iou),         # IoU of the selected LCC against GT
                    "dice_lcc": float(dice_lcc),
                    "iou": iou_full,
                    "dice": dice_full,
                    "dlr": dlr,
                    "dbr": dbr,
                    "precision": precision,
                    "leakage": leakage,
                    "amr": amr,
                }
                self.test_step_outputs.append(metrics)
                filename = os.path.basename(item.get("image_meta_dict", {}).get("filename_or_obj", f"unknown_{batch_idx}_{i}"))
                print(f"Metrics for {filename}: {metrics}")
                # 1a) Optional, if we want to save during metric testing (invert + postproc + save)
                if getattr(args, "save_val", False):
                    invert_source_transform = self.data_module.val_transforms
                    keys_to_invert = ["pred"]
                    meta_keys_list = ["pred_meta_dict"]
                    if "lung" in item and item["lung"] is not None:
                        keys_to_invert.append("lung")
                        meta_keys_list.append("lung_meta_dict")

                    inverter = Compose([
                        Invertd(
                            keys=keys_to_invert,
                            transform=invert_source_transform,
                            orig_keys="image",
                            meta_keys=meta_keys_list,
                            orig_meta_keys="image_meta_dict",
                            meta_key_postfix="meta_dict",
                            nearest_interp=True,
                            to_tensor=True,
                        )
                    ])  # documented path for invertible pipelines.
                    save_item = inverter(dict(item))

                    # Ensure binary after inversion, still using `thr`
                    save_item = Compose([AsDiscreted(keys="pred", threshold=thr)])(save_item)

                    # ----- Optional ROI gating / LCC -----
                    try:
                        import numpy as _np
                        from scipy.ndimage import binary_dilation as _bin_dilate

                        pred_np = (save_item["pred"].cpu().numpy().squeeze() > thr).astype(_np.uint8)
                        initial_pred_count = int(_np.sum(pred_np > 0))
                        roi_mode = os.getenv("LUNG_POSTPROC_MODE", "off").lower()
                        k = int(os.getenv("KEEP_COMPONENTS", "2"))
                        apply_pp = (roi_mode in ("mask", "bbox")) or (k > 0)

                        if apply_pp:
                            lung_np = None
                            if "lung" in save_item and save_item["lung"] is not None:
                                lung_np = (save_item["lung"].cpu().numpy().squeeze() > 0).astype(_np.uint8)
                                if lung_np.shape != pred_np.shape:
                                    lung_np = None

                            if roi_mode in ("mask", "bbox") and (lung_np is not None) and float(_np.sum(lung_np > 0)) > 0.05 * float(_np.prod(lung_np.shape)):
                                # spacing from affine
                                _vx = 1.0
                                if hasattr(save_item["pred"], "affine") and save_item["pred"].affine is not None:
                                    _aff = save_item["pred"].affine
                                    _aff = _aff.detach().cpu().numpy() if hasattr(_aff, "detach") else _np.array(_aff)
                                    _spacing = _np.sqrt((_aff[:3, :3] ** 2).sum(axis=0)).astype(float)
                                    x_sp = float(_spacing[0]) if _spacing.size >= 1 else 1.0
                                    if not _np.isfinite(x_sp) or x_sp <= 0.0:
                                        x_sp = float(_np.mean(_spacing)) if _np.isfinite(_np.mean(_spacing)) else 1.0
                                    _vx = x_sp

                                if roi_mode == "mask":
                                    _mm = float(os.getenv("LUNG_MASK_DILATION_MM", "10.0"))
                                    _iters = int(max(0, round(_mm / float(max(_vx, 1e-6)))))
                                    lung_dilated = _bin_dilate(lung_np > 0, iterations=_iters)
                                    test_masked = pred_np * lung_dilated.astype(_np.uint8)
                                    masked_count = int(_np.sum(test_masked > 0))
                                    removal_ratio = (initial_pred_count - masked_count) / float(max(initial_pred_count, 1))
                                    if removal_ratio <= float(os.getenv("LUNG_MASK_MAX_REMOVAL", "0.5")):
                                        pred_np = test_masked
                                else:  # bbox
                                    _mm_margin = float(os.getenv("LUNG_BBOX_MARGIN_MM", "5.0"))
                                    _vox_margin = int(max(0, round(_mm_margin / float(max(_vx, 1e-6)))))
                                    zs, ys, xs = _np.where(lung_np > 0)
                                    if zs.size > 0:
                                        z0, z1 = max(0, int(zs.min()) - _vox_margin), min(pred_np.shape[0]-1, int(zs.max()) + _vox_margin)
                                        y0, y1 = max(0, int(ys.min()) - _vox_margin), min(pred_np.shape[1]-1, int(ys.max()) + _vox_margin)
                                        x0, x1 = max(0, int(xs.min()) - _vox_margin), min(pred_np.shape[2]-1, int(xs.max()) + _vox_margin)
                                        crop_mask = _np.zeros_like(pred_np, dtype=_np.uint8)
                                        crop_mask[z0:z1+1, y0:y1+1, x0:x1+1] = 1
                                        pred_np = (pred_np > 0).astype(_np.uint8) & crop_mask

                            # LCC (dictionary version) — documented post transform.
                            if k > 0 and pred_np.any():
                                tmp = {"pred": torch.from_numpy(pred_np[None])}
                                tmp = KeepLargestConnectedComponentd(keys="pred", num_components=k)(tmp)
                                pred_np = tmp["pred"].cpu().numpy().squeeze().astype(np.uint8)
                            _old = save_item["pred"]
                            pred_t = torch.from_numpy(pred_np[None]).to(device=_old.device, dtype=_old.dtype)
                            save_item["pred"] = MetaTensor(pred_t, meta=getattr(_old, "meta", None), affine=getattr(_old, "affine", None))

                    except Exception as _e:
                        print(f"  Warning: post-inversion processing during metric-saving failed: {_e}")

                    if "lung" in save_item:
                        save_item.pop("lung")

                    saver = SaveImageWithOriginalMetadatad(
                        keys="pred", output_dir=os.path.join(output_path, "preds"),
                        output_postfix="seg", separate_folder=True, print_log=True,
                        resample_full_volume=True,
                    )
                    saver(save_item)
                # move to next item
                continue
            # 2) Invert only the prediction back to original space
            use_val_chain = bool(getattr(args, "metric_testing", False)) or (
                str(getattr(args, "test_dataset", "")).lower() == "val"
            )
            invert_source_transform = (
                self.data_module.val_transforms if use_val_chain else self.data_module.test_transforms
            )
            keys_to_invert = ["pred"]
            meta_keys_list = ["pred_meta_dict"]
            if "lung" in item and item["lung"] is not None:
                keys_to_invert.append("lung")
                meta_keys_list.append("lung_meta_dict")

            inverter = Compose([
                Invertd(
                    keys=keys_to_invert,
                    transform=invert_source_transform,
                    orig_keys="image",
                    meta_keys=meta_keys_list,
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=True,
                    to_tensor=True,
                    )
            ])
            processed_item = inverter(item)
            processed_item = Compose([AsDiscreted(keys="pred", threshold=float(os.getenv("PRED_THRESHOLD", getattr(args, "pred_threshold", 0.5))))])(processed_item)
            # 2b) Optional post-inversion lung ROI gating (mask | bbox) and LCC for inference
            # Optional ROI gating / LCC in inference mode
            roi_mode = os.getenv("LUNG_POSTPROC_MODE", "off").lower()
            k = int(os.getenv("KEEP_COMPONENTS", "2"))
            apply_pp = (roi_mode in ("mask", "bbox")) or (k > 0)
            if apply_pp:
                try:
                    import numpy as _np
                    from scipy.ndimage import binary_dilation as _bin_dilate

                    pred_np = (processed_item["pred"].cpu().numpy().squeeze() > float(os.getenv("PRED_THRESHOLD", getattr(args, "pred_threshold", 0.5)))).astype(_np.uint8)
                    initial_pred_count = int(_np.sum(pred_np > 0))
                    print(f"  Initial prediction voxels: {initial_pred_count:,}")

                    lung_np = None
                    if roi_mode in ("mask", "bbox"):
                        if "lung" in processed_item and processed_item["lung"] is not None:
                            lung_np = (processed_item["lung"].cpu().numpy().squeeze() > 0).astype(_np.uint8)
                            if lung_np.shape != pred_np.shape:
                                print(f"  WARNING: Lung shape {lung_np.shape} != pred shape {pred_np.shape}")
                                lung_np = None

                    if (roi_mode in ("mask", "bbox")) and (lung_np is not None) and float(_np.sum(lung_np > 0)) > 0.05 * float(_np.prod(lung_np.shape)):
                        try:
                            _vx = 1.0
                            if hasattr(processed_item["pred"], "affine") and processed_item["pred"].affine is not None:
                                _aff = processed_item["pred"].affine
                                _aff = _aff.detach().cpu().numpy() if hasattr(_aff, "detach") else _np.array(_aff)
                                _spacing = _np.sqrt((_aff[:3, :3] ** 2).sum(axis=0)).astype(float)
                                x_sp = float(_spacing[0]) if _spacing.size >= 1 else 1.0
                                if not _np.isfinite(x_sp) or x_sp <= 0.0:
                                    x_sp = float(_np.mean(_spacing)) if _np.isfinite(_np.mean(_spacing)) else 1.0
                                _vx = x_sp

                            if roi_mode == "mask":
                                _mm = float(os.getenv("LUNG_MASK_DILATION_MM", "10.0"))
                                _iters = int(max(0, round(_mm / float(max(_vx, 1e-6)))))
                                lung_dilated = _bin_dilate(lung_np > 0, iterations=_iters)
                                test_masked = pred_np * lung_dilated.astype(_np.uint8)
                                masked_count = int(_np.sum(test_masked > 0))
                                removal_ratio = (initial_pred_count - masked_count) / float(max(initial_pred_count, 1))
                                print(f"  Lung masking would remove {removal_ratio:.1%} of prediction")
                                if removal_ratio <= float(os.getenv("LUNG_MASK_MAX_REMOVAL", "0.5")):
                                    pred_np = test_masked
                                    print(f"  Applied lung masking (removed {removal_ratio:.1%})")
                                else:
                                    print(f"  SKIPPED lung masking - would remove too much ({removal_ratio:.1%})")
                            else:  # bbox
                                _mm_margin = float(os.getenv("LUNG_BBOX_MARGIN_MM", "5.0"))
                                _vox_margin = int(max(0, round(_mm_margin / float(max(_vx, 1e-6)))))
                                zs, ys, xs = _np.where(lung_np > 0)
                                if zs.size > 0:
                                    z0, z1 = max(0, int(zs.min()) - _vox_margin), min(pred_np.shape[0]-1, int(zs.max()) + _vox_margin)
                                    y0, y1 = max(0, int(ys.min()) - _vox_margin), min(pred_np.shape[1]-1, int(ys.max()) + _vox_margin)
                                    x0, x1 = max(0, int(xs.min()) - _vox_margin), min(pred_np.shape[2]-1, int(xs.max()) + _vox_margin)
                                    crop_mask = _np.zeros_like(pred_np, dtype=_np.uint8)
                                    crop_mask[z0:z1+1, y0:y1+1, x0:x1+1] = 1
                                    pred_np = (pred_np > 0).astype(_np.uint8) & crop_mask

                        except Exception as e:
                            print(f"  Lung ROI gating failed: {e}, proceeding without ROI gating")

                    # Keep top-K components if requested
                    if k > 0 and pred_np.any():
                        tmp = {"pred": torch.from_numpy(pred_np[None])}
                        tmp = KeepLargestConnectedComponentd(keys="pred", num_components=k)(tmp)
                        pred_np = tmp["pred"].cpu().numpy().squeeze().astype(np.uint8)

                    # WRITE BACK and preserve meta/affine
                    _old = processed_item["pred"]
                    pred_t = torch.from_numpy(pred_np[None]).to(device=_old.device, dtype=_old.dtype)
                    processed_item["pred"] = MetaTensor(pred_t, meta=getattr(_old, "meta", None), affine=getattr(_old, "affine", None))

                    if getattr(args, "debug_inference", False):
                        try:
                            import nibabel as _nib
                            aff_post = processed_item["pred"].affine.cpu().numpy() if hasattr(processed_item["pred"], "affine") else _np.eye(4)
                            _nib.save(_nib.Nifti1Image(pred_np.astype(_np.uint8), aff_post), os.path.join(debug_dir, f"{base_name}_4_post_invert_topk.nii.gz"))
                        except Exception as e:
                            print(f"  Debug save failed: {e}")

                except Exception as e:
                    print(f"  Warning: post-inversion processing failed: {e}")
            else:
                print("Metric testing mode detected. Skipping post-inversion processing.")

            # Debug after inversion
            print(f"  Prediction shape after inversion: {processed_item['pred'].shape}")
            pred_voxels = torch.sum(processed_item["pred"] > 0).item()
            print(f"  Non-zero prediction voxels after inversion: {pred_voxels}")

            # Save final prediction
            if "lung" in processed_item:
                processed_item.pop("lung")
            saver = SaveImageWithOriginalMetadatad(
                keys="pred",
                output_dir=os.path.join(output_path, "preds"),
                output_postfix="seg",
                separate_folder=True,
                print_log=True,
                resample_full_volume=True,
            )
            saver(processed_item)

        del tst_inputs, logits, decollated_batch, decollated_logits
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        if not args.metric_testing:
            print("Inference complete. No metrics calculated.")
            return

        if not self.test_step_outputs:
            print("No test step outputs found. Saving empty metrics file.")
            # Still save an empty file to indicate completion.
            metrics_file = os.path.join(output_path, "final_test_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump({}, f, indent=4)
            return
            
        avg_metrics = {}
        metric_keys = self.test_step_outputs[0].keys() if self.test_step_outputs else []
        for key in metric_keys:
            valid_values = [d[key] for d in self.test_step_outputs if key in d and d[key] is not None]
            if valid_values:
                avg_metrics[f"avg_test_{key}"] = np.mean(valid_values)
            else:
                avg_metrics[f"avg_test_{key}"] = 0.0 # Or float('nan')

        # Also export Dice loss (1 - mean dice)
        if 'dice_full' in metric_keys:
            avg_metrics['avg_test_dice_loss'] = 1.0 - avg_metrics.get('avg_test_dice_full', 0.0)
        elif 'dice' in metric_keys:
            avg_metrics['avg_test_dice_loss'] = 1.0 - avg_metrics.get('avg_test_dice', 0.0)
        elif 'iou' in metric_keys or 'iou_full' in metric_keys:
             # If only IoU is present, still include a placeholder dice loss (cannot derive exactly from IoU)
             avg_metrics['avg_test_dice_loss'] = None

        # Log metrics for PyTorch Lightning to collect
        for key, value in avg_metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Also save detailed metrics to a JSON file for inspection
        detailed_metrics_file = os.path.join(output_path, "detailed_test_metrics.json")
        with open(detailed_metrics_file, 'w') as f:
            json.dump(self.test_step_outputs, f, indent=4)
        print(f"Detailed per-case test metrics saved to {detailed_metrics_file}")
        
        # Save the final averaged metrics to the file the user expects
        final_metrics_file = os.path.join(output_path, "final_test_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        print(f"Final aggregated test metrics saved to {final_metrics_file}")
            
    def generate_validation_predictions(self):
        """Generate predictions on the validation split of the training data."""
        print("Generating predictions on validation split...")
        val_loader = self.data_module.aff_training_val_dataloader()
        
        # Create output directory
        val_pred_dir = os.path.join(output_path, "val_preds")
        os.makedirs(val_pred_dir, exist_ok=True)
        
        # Save validation predictions
        val_post_processing = Compose([
            DebugMetadatad(keys=["pred"], prefix="Before post-processing"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            CalculateMetricsd(keys="pred", label_key="label"),
            SaveImageWithOriginalMetadatad(
                keys="pred",
                output_dir=val_pred_dir,
                output_postfix="seg",
                separate_folder=True,
                print_log=True
            ),
        ])
        
        results = []
        
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch["image"].to(self._device)
            
            with torch.no_grad():
                if batch_idx % 5 == 0:
                    print(f"Processing validation batch {batch_idx}/{len(val_loader)}")
                
                roi_size = (args.test_patch_size if args.test_patch_size is not None else args.patch_size)
                if isinstance(roi_size, int):
                    roi_size = (roi_size,) * 3
                
                # Run inference
                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=self.forward,
                    overlap=0.5,
                    mode="gaussian",
                    device=self._device,
                    progress=True,
                ).cpu()
                
                # Process each item in the batch
                images = decollate_batch(batch["image"])
                labels = decollate_batch(batch["label"]) if "label" in batch else [None] * len(images)
                
                for i, (img_mt, lbl_mt) in enumerate(zip(images, labels)):
                    pred_mt = MetaTensor(logits[i])
                    pred_mt.meta = img_mt.meta.copy()
                    
                    item = {"image": img_mt, "pred": pred_mt}
                    if lbl_mt is not None:
                        item["label"] = lbl_mt
                    
                    out_list = val_post_processing([item])
                    if out_list and "metrics" in out_list[0]:
                        results.append(out_list[0]["metrics"])
            
            # Clear memory
            del inputs, logits
            torch.cuda.empty_cache()
        
        # Calculate average metrics
        if results:
            print("Validation prediction results:")
            avg_metrics = {}
            metric_keys = results[0].keys()
            for key in metric_keys:
                avg_metrics[key] = np.mean([d[key] for d in results if key in d])
                print(f"Average {key}: {avg_metrics[key]:.4f}")
            
            # Save metrics to CSV
            metrics_file = os.path.join(val_pred_dir, "val_metrics.csv")
            with open(metrics_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(avg_metrics.keys())
                writer.writerow(avg_metrics.values())
            print(f"Validation metrics saved to {metrics_file}")


if __name__ == '__main__':
    callbacks = []
    try:
        # Force CUDA initialization and setup
        if torch.cuda.is_available():
            print("=== CUDA is available ===")
            #CUDA initialization
            torch.tensor([1.0], device='cuda')
            torch.cuda.synchronize()
            print(f"CUDA initialized successfully.")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("=== CUDA is NOT available ===")
        
        # Ensure multiprocessing start method is safe for CUDA
        if params_dict['NUM_WORKERS'] > 0:
            current_method = mp.get_start_method(allow_none=True)
            desired = 'spawn' if torch.cuda.is_available() else 'fork'
            if current_method != desired:
                try:
                    mp.set_start_method(desired, force=True)
                    print(f"Set multiprocessing start method to '{desired}'")
                except RuntimeError as e:
                    print(f"Could not set start method to '{desired}': {e}. Using '{current_method}'.")
            else:
                print(f"Multiprocessing start method already '{current_method}'.")

        model = SegmentationNet(device=device, data_module=data_module)
        model._initialize_weights()
        
        # Configure PyTorch for better dataloader worker handling
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        data_module = SegmentationDataModule(
            data_path=data_path_for_datamodule,
            batch_size=args.batch_size,
            dataset=args.dataset,
            params=params_dict,
            compute_dtype=torch.float16 if args.mixed else torch.float32,
            args=args
        )
        
        # Create SmartCacheCallback conditionally - only if SmartCache is enabled
        smartcache_callback = None
        if params_dict.get('SMARTCACHE', False):
            data_module.setup("fit") # Ensure datasets are prepared
            if hasattr(data_module, 'train_ds') and data_module.train_ds is not None:
                actual_smartcache_ds = None
                if hasattr(data_module.train_ds, 'data') and \
                   isinstance(data_module.train_ds.data, monai.data.SmartCacheDataset):
                    actual_smartcache_ds = data_module.train_ds.data
                    print(f"Found SmartCacheDataset instance (data_module.train_ds.data): {type(actual_smartcache_ds)}")
                
                if actual_smartcache_ds is not None:
                    data_module.smartcache_dataset_train = actual_smartcache_ds 
                    smartcache_callback = SmartCacheCallback(actual_smartcache_ds)
                    callbacks.append(smartcache_callback)
                    print(f"SmartCacheCallback created with dataset type: {type(actual_smartcache_ds)} and added to callbacks")
                else:
                    print("WARNING: SmartCache was requested, but SmartCacheDataset instance not found in data_module.train_ds.data. Callback not added.")
            else:
                print("WARNING: SmartCache was requested, but data_module.train_ds is None after setup. Callback not added.")
        elif params_dict.get('CACHE_DATASET', False):
            print("Using CacheDataset only - SmartCacheCallback not needed")
        else:
            print("WARNING: SmartCacheCallback not used because caching is disabled")
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_path, # This will be overridden if trainer specifies default_root_dir
            monitor="val_iou", # Monitor the aggregated epoch IoU
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False, # Keep false if custom filename
            filename="best_metric_model_{epoch:02d}-{val_iou:.4f}", # Use val_iou_epoch in filename
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop_callback = EarlyStopping(
            monitor="val_iou", # Monitor the aggregated epoch IoU
            min_delta=0.001, 
            patience=20, 
            verbose=False, 
            mode="max"
        )
        callbacks.extend([checkpoint_callback, lr_monitor, early_stop_callback])
        
        if args.training:
            # only instantiate once and move to GPU
            net = SegmentationNet(device=device, data_module=data_module)
            net.to(device)
            # set up checkpoints
            model_path = os.path.join(output_path, "model")
            os.makedirs(model_path, exist_ok=True)
            # Re-define checkpoint_callback with the correct model_path for trainer
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_path, # Specific path for training checkpoints
                monitor="val_iou", # Monitor the aggregated epoch IoU
                mode="max",
                save_top_k=3,
                save_last=True,
                auto_insert_metric_name=False,
                filename="best_metric_model_{epoch:02d}-{val_iou:.4f}", # Use val_iou
            )
            # Update callbacks list for the trainer (lr_monitor and early_stop_callback are already defined correctly)
            current_callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
            if smartcache_callback is not None: # Only add if successfully created
                current_callbacks.append(smartcache_callback)
                print("SmartCacheCallback added to trainer callbacks.")
            # Fallback check if smartcache_callback wasn't created above but was intended
            elif params_dict.get('SMARTCACHE', False) and hasattr(data_module, 'train_ds') and data_module.train_ds is not None:
                 # Check if we can find the SmartCacheDataset in the proper location
                 if hasattr(data_module.train_ds, 'data') and data_module.train_ds.data is not None:
                    actual_smartcache_ds = data_module.train_ds.data
                    if isinstance(actual_smartcache_ds, monai.data.SmartCacheDataset):
                        data_module.smartcache_dataset_train = actual_smartcache_ds
                        smartcache_callback = SmartCacheCallback(actual_smartcache_ds)
                        current_callbacks.append(smartcache_callback)
                        print(f"SmartCacheCallback created (fallback) with dataset type: {type(actual_smartcache_ds)} and added to trainer callbacks.")
                    else:
                        print(f"WARNING: data_module.train_ds.data exists but is not a SmartCacheDataset (type: {type(actual_smartcache_ds)}).")
                 else:
                    print("WARNING: SmartCache enabled but data_module.train_ds.data is not accessible.")

            trainer_callbacks = current_callbacks # Use the updated list for the trainer

            print(f"The output path is {output_path}")
            print(f"The model_path for training checkpoints is {model_path}")
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                num_nodes=1,
                max_epochs=args.max_epochs,
                default_root_dir=model_path, # Trainer will use ts for its own logging if needed
                log_every_n_steps=20,
                precision='16-mixed' if args.mixed else '32-true',
                strategy="auto",
                enable_model_summary=True,
                max_time=(datetime.timedelta(hours=args.max_time_hours) if (args.max_time_hours is not None and args.max_time_hours > 0) else None),
                deterministic="warn",
                gradient_clip_val=1.0,
                num_sanity_val_steps=1, 
                enable_checkpointing=True, 
                accumulate_grad_batches=1,
                check_val_every_n_epoch=1, 
                enable_progress_bar=True,
                sync_batchnorm=False, 
                callbacks=trainer_callbacks,
            )
            print("\n==== GPU INFORMATION BEFORE TRAINING ====")
            monitor_gpu_usage()   
            
            ckpt_path = None
            if args.resume_from_checkpoint:
                if os.path.exists(args.resume_from_checkpoint):
                    ckpt_path = args.resume_from_checkpoint
                    print(f"Resuming training from checkpoint: {ckpt_path}")
                else:
                    print(f"Warning: Checkpoint path {args.resume_from_checkpoint} does not exist. Starting fresh training.")
            
            # Optional weights-only fine-tune (do NOT resume optimizer/scheduler)
            if not ckpt_path and args.finetune_from:
                if os.path.exists(args.finetune_from):
                    try:
                        checkpoint = torch.load(args.finetune_from, map_location=device)
                        state_dict = checkpoint.get("state_dict", checkpoint)
                        missing, unexpected = net.load_state_dict(state_dict, strict=False)
                        print(f"Loaded weights from {args.finetune_from} (weights-only). Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                        if len(missing) > 0:
                            print(f"First few missing keys: {missing[:10]}")
                        if len(unexpected) > 0:
                            print(f"First few unexpected keys: {unexpected[:10]}")
                    except Exception as e:
                        print(f"Failed to load weights from {args.finetune_from}: {e}")
                else:
                    print(f"Warning: --finetune_from path not found: {args.finetune_from}")

            trainer.fit(net, datamodule=data_module, ckpt_path=ckpt_path)
            print(f"train completed, best_metric: {net.best_val_dice:.4f} Dice and {net.best_val_iou:.4f} IoU " f"at epoch {net.best_val_epoch}")
            print("\n==== GPU INFORMATION AFTER TRAINING ====")
            monitor_gpu_usage()

            best_model_path = checkpoint_callback.best_model_path
            print(f"Best model saved at: {best_model_path}")
            print("\n=== AUTOMATICALLY STARTING TESTING ON VALIDATION SPLIT OF TRAINING DATA ===")
            
            test_net = SegmentationNet(device=device, data_module=data_module)
            
            if best_model_path and os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device)
                test_net.load_state_dict(checkpoint["state_dict"])
                print(f"Loaded best model from: {best_model_path}")
                
                # Move model to GPU
                test_net.to(device)
                diagnose_model_placement(test_net._model)
                
                # Create a trainer for testing
                trainer_test = pl.Trainer(
                    devices=1,
                    accelerator="gpu",
                    logger=False,
                    enable_model_summary=False,
                    enable_progress_bar=True,
                    precision='16-mixed' if args.mixed else '32-true',
                )
                #This ensures that test_dataloader() and test_transforms are ready.
                print("Setting up datamodule for testing stage...")
                data_module.setup(stage='test')
                gc.collect()
                torch.cuda.empty_cache()
                print("Running test on validation split of training data...")
                print(f"Validation split dataset size: {len(data_module.val_files)} samples")
                
                print(f"DataModule val_files: {len(data_module.val_files)}")
                data_module.setup()  # Make sure setup is called
                print(f"After setup - val_files: {len(data_module.val_files)}")
                
                test_results = trainer_test.test(test_net, datamodule=data_module)
                print("Testing completed!")
                print(f"Test results: {test_results}")
                
                # Save test results
                test_results_file = os.path.join(output_path, "test_results.json")
                with open(test_results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"Test results saved to: {test_results_file}")
                
                del test_net
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Warning: Could not find best model checkpoint at {best_model_path}")
                print("Skipping automatic testing phase.")

        if args.validation:
            # initialize the lightning module
            net = SegmentationNet(device=device, data_module=data_module)
            net.to(device)
            diagnose_model_placement(net._model)
            model_path = os.path.join(output_path, "model_AWC")
            os.makedirs(model_path, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                dirpath=model_path,
                monitor="val_iou", # Monitor the aggregated epoch IoU
                mode="max",
                save_top_k=3 if not args.awc else -1,
                save_last=True,
                auto_insert_metric_name=False,
                filename="best_metric_model_{epoch:02d}-{val_iou:.4f}" if not args.awc else "best_metric_awc_{epoch:02d}-{val_iou:.4f}",
            )
            # lr_monitor is already defined
            early_stop_callback = EarlyStopping(
                monitor="val_iou", # Monitor the aggregated epoch IoU
                min_delta=0.002, 
                patience=50, 
                verbose=False, 
                mode="max"
            )
            
            validation_callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
            if 'smartcache_callback' in locals() and smartcache_callback is not None and data_module.cache_dataset and data_module.use_smartcache:
                validation_callbacks.append(smartcache_callback)


            # initialize the trainer
            trainer = pl.Trainer(
                devices=1,
                accelerator="gpu",
                num_nodes=1,
                callbacks=validation_callbacks, # Use the updated list
                max_epochs=0 if not args.awc else args.max_epochs,
                default_root_dir=model_path,
                log_every_n_steps=5,
                gradient_clip_val=1.0,
                precision='16-mixed' if args.mixed else '32-true',
                enable_model_summary=True,
                num_sanity_val_steps=1,
            )
            net.load_state_dict(torch.load(best_model_path)["state_dict"])
            # net.load_from_checkpoint(checkpoint_path=best_model_path)
            trainer.fit(net, datamodule=data_module)
            trainer.test(net, ckpt_path=best_model_path if not args.awc else "best")
        
        if args.testing or args.metric_testing:
            print("=== STARTING TESTING PHASE ===")
            net = SegmentationNet(device=device, data_module=data_module)
            
            # Construct path to the best model from training phase
            model_dir = os.path.join(output_path, "model")
            
            checkpoint_path = None
            if args.weights_path:
                if os.path.exists(args.weights_path):
                    checkpoint_path = args.weights_path
                    print(f"Using explicitly provided checkpoint: {checkpoint_path}")
                else:
                    candidate = os.path.join(model_dir, os.path.basename(args.weights_path))
                    if os.path.exists(candidate):
                        checkpoint_path = candidate
                        print(f"Using checkpoint resolved under model dir: {checkpoint_path}")
                    else:
                        print(f"Provided --weights_path not found: {args.weights_path}. Will fall back to latest in {model_dir}.")
            else:
                print(f"No --weights_path provided. Will fall back to latest in {model_dir}.")
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                net.load_state_dict(checkpoint["state_dict"])
                print("Model loaded successfully!")
                net.to(device)
            else:
                print(f"ERROR: Checkpoint not found at {checkpoint_path}")
                sys.exit(1)
            
            # Create a trainer for testing
            trainer_test = pl.Trainer(
                devices=1,
                accelerator="gpu",
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=True,
                precision='16-mixed' if args.mixed else '32-true',
            )
            
            # Setup datamodule for the 'test' stage (which uses the official validation set)
            print("Setting up datamodule for 'test' stage (using official validation data)...")
            data_module.setup(stage='test') 
            
            test_results = trainer_test.test(net, datamodule=data_module)
            print("Testing completed!")
            
            # Save final aggregate results
            test_results_file = os.path.join(output_path, "final_test_metrics.json")
            with open(test_results_file, 'w') as f:
                json.dump(test_results[0] if test_results else {}, f, indent=4)
            print(f"Final aggregated test results saved to: {test_results_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        if torch.cuda.is_available():
            print("GPU diagnostics after error:")
            monitor_gpu_usage()
        raise e
