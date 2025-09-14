import pytorch_lightning as pl
import numpy as np
from glob import glob
import os
import torch
import hashlib
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    RandCropByPosNegLabeld,
    CenterSpatialCropd,
    SpatialPadd,
    ResizeWithPadOrCropd,
)
from monai.transforms.intensity.dictionary import (
    ScaleIntensityRanged,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    NormalizeIntensityd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import (
    Spacingd,
    RandFlipd,
    RandRotate90d,
    Orientationd,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utility.dictionary import ToTensord, CastToTyped, EnsureTyped, Lambdad
from monai.data.dataset import (
    Dataset,
    PersistentDataset,
    SmartCacheDataset,
    CacheDataset,
)
from monai.transforms import (EnsureChannelFirstd,)
from monai.data.dataloader import DataLoader
import random
from .utils import get_folder_names
import gc
from pathlib import Path
from monai.data.meta_tensor import MetaTensor
from scipy.ndimage import binary_dilation, convolve, label as ndimage_label
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Optional, Dict
from monai.data.utils import pad_list_data_collate
from monai.utils.misc import ensure_tuple
from monai.data.image_reader import NumpyReader
import json, os
# Set random seed
seed_num = 123
random.seed(seed_num)

# Safe globals for multiprocessing
from torch.serialization import add_safe_globals
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys
import numpy.core.multiarray as multiarray

safe_globals_to_add = [
    MetaKeys, MetaTensor, SpaceKeys, TraceKeys,
    np.ndarray, np.dtype, torch.Size, torch.Tensor,
    tuple, list, dict, type(None),
]
add_safe_globals(safe_globals_to_add)

import SimpleITK as sitk
import numpy as np
import torch
from monai.transforms import MapTransform
from monai.data import MetaTensor
import os

from seg.utils import find_file_in_path, get_folder_names

# NEW: optional patch bank sampler
try:
    from .patch_bank_sampler import SamplePatchFromBankd
except Exception as e:
    import traceback
    print(f"[DataModule] patch-bank import failed: {e}\n{traceback.format_exc()}")
    SamplePatchFromBankd = None


class SetSpacingFromMeta(MapTransform):
    """
    Sets the spacing of a MetaTensor based on a corresponding _meta.json file.
    It modifies the affine matrix of the MetaTensor to reflect the true spacing of the pre-processed data.
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        # We only need to act on the 'image' key, as it's the reference for spacing.
        key = 'image'
        if key not in self.key_iterator(d):
            return d
        # Get the item
        item = d.get(key)
        if not isinstance(item, MetaTensor) or not hasattr(item, 'meta'):
            return d
        
        npy_filepath_str = item.meta.get('filename_or_obj')
        if not npy_filepath_str:
            return d
        
        json_path = Path(str(npy_filepath_str).replace('.npy', '_meta.json'))

        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    meta_info = json.load(f)
                # Extract spacing, direction, and origin
                spacing = meta_info.get("processed_spacing")
                direction = np.array(meta_info.get("processed_direction", [1,0,0,0,1,0,0,0,1])).reshape(3,3)
                origin = np.array(meta_info.get("processed_origin", [0,0,0]))
                
                if spacing:
                    # Construct the new affine matrix
                    affine = np.eye(4)
                    affine[:3, :3] = direction @ np.diag(spacing)
                    affine[:3, 3] = origin
                    # Move to the same device as the item
                    # Affine matrices in MONAI are typically float64.
                    new_affine = torch.from_numpy(affine.astype(np.float64)).to(device=item.device)
                    # Update the affine for the image and propagate to other keys
                    for k in self.key_iterator(d):
                        if isinstance(d.get(k), MetaTensor):
                            d[k].affine = new_affine
            except Exception as e:
                print(f"Warning: [SetSpacingFromMeta] Failed to process {json_path}: {e}")
        return d

    def __repr__(self):
        return f"SetSpacingFromMeta(keys={self.keys})"


class BinarizeLungd(MapTransform):
    """
    Top-level transform to binarize lung mask(s) to {0,1} without using non-picklable lambdas.
    This avoids multiprocessing pickling errors in DataLoader workers.
    """
    def __init__(self, keys, allow_missing_keys: bool = True):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and d[key] is not None:
                arr = d[key]
                try:
                    if isinstance(arr, MetaTensor):
                        d[key] = (arr > 0).to(arr.dtype)
                    elif isinstance(arr, torch.Tensor):
                        d[key] = (arr > 0).to(arr.dtype)
                    else:
                        d[key] = (np.asarray(arr) > 0).astype(arr.dtype)
                except Exception:
                    # Best-effort fallback
                    d[key] = (np.asarray(arr) > 0).astype(np.uint8)
        return d

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


def worker_init_fn_enhanced(worker_id):
    """Enhanced worker initialization for multiprocessing."""
    import torch
    import numpy as np
    import random
    import os
    
    # Set unique seeds for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # Optimize for multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class LoadOriginalNiftiHeaderd(MapTransform):
    """
    This transform class is designed to load the header from the original NIfTI file
    before any resampling or spacing adjustments have been made. It retrieves the
    affine transformation matrix from the file's metadata and stores it in the
    data dictionary under the key `f"{key}_original_affine"`.
    This is crucial for inverting transformations post-inference to ensure predictions
    are aligned with the original image's coordinate system.
    """
    def __init__(self, keys, allow_missing_keys: bool = True):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                meta_dict = d.get(f"{key}_meta_dict", d[key].meta if isinstance(d[key], MetaTensor) else {})
                filepath = meta_dict.get("filename_or_obj")

                if filepath:
                    # Handles cases where filepath might be a list of paths
                    if isinstance(filepath, list):
                        filepath = filepath[0] if filepath else None
                    
                    if filepath:
                        filepath_str = "./" + str(filepath)
                    original_filepath = filepath_str.replace(".npy", ".nii.gz")

                    # Special handling for validation files
                    if 'val/' in original_filepath:
                        original_filepath = original_filepath.replace('val/', 'imagesVal/')

                    if os.path.exists(original_filepath):
                        try:
                            # Load the original NIfTI file using SimpleITK, which is more robust
                            img = sitk.ReadImage(original_filepath)
                            direction = np.array(img.GetDirection()).reshape(3, 3)
                            spacing = np.array(img.GetSpacing())
                            origin = np.array(img.GetOrigin())
                            # Construct the affine matrix
                            affine = np.eye(4)
                            affine[:3, :3] = direction @ np.diag(spacing)
                            affine[:3, 3] = origin
                            # Store the original affine and shape in the dictionary
                            d[f"{key}_original_affine"] = torch.from_numpy(affine).to(torch.float64)
                            d[f"{key}_original_shape"] = torch.tensor(img.GetSize(), dtype=torch.int32)

                            if isinstance(d[key], MetaTensor):
                                d[key].meta['original_affine'] = d[f"{key}_original_affine"]
                                d[key].meta['original_spatial_shape'] = d[f"{key}_original_shape"]

                        except Exception as e:
                            print(f"Warning: Could not read header for {original_filepath}: {e}")
                    else:
                        print(f"Warning: Original NIfTI file not found at {original_filepath}, cannot load header.")
        return d

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


class SegmentationDataModule(pl.LightningDataModule):
    """
    DataModule optimized for CPU/GPU transform efficiency and large images (512x512xH).
    """

    def __init__(self, data_path, dataset="ATM22", batch_size=1, test_batch_size=1,
                train_val_ratio=0.85, augment=True, params=None, compute_dtype=torch.float32, args=None):
        super().__init__()
        self.data_path = data_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_val_ratio = train_val_ratio
        self.do_augment = augment
        self.params = params or {}
        self.compute_dtype = compute_dtype
        self.args = args # Stores args to control dataloader logic
        self._setup_called = False # Guards to prevent multiple setup calls
        
        # Memory-optimized settings for large images
        self.num_workers = min(int(self.params.get("NUM_WORKERS", 2)), 4)  # Limit workers for large images
        self.prefetch_factor = int(self.params.get("PREFETCH_FACTOR", 1))  # Reduce prefetch for memory
        self.pin_memory = bool(self.params.get("PIN_MEMORY", False))  # Disable for large images
        
        # PersistentDataset settings
        self.cache_dataset = bool(self.params.get("CACHE_DATASET", False))
        # Use dataset-root persistent cache so it aligns with cleanup in scripts
        # Example: ./ATM22/npy_files/persistent_cache
        self.cache_dir = Path(self.data_path) / "persistent_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_memory_mb = int(self.params.get("MAX_CACHE_MEMORY_MB", 16000))  # 16GB default

        # Disk-backed caching (safe on RAM) using PersistentDataset
        self.use_persistent_dataset = bool(self.params.get("PERSISTENT_DATASET", True))
        print(f"use_persistent_dataset: {self.use_persistent_dataset}")
        # Transform optimization settings
        self.use_gpu_transforms = bool(self.params.get("USE_GPU_TRANSFORMS", False)) 
        self.cache_rate = float(self.params.get("CACHE_RATE", 0.5))
        # Dataset configurations
        self.patch_size = self.params.get("PATCH_SIZE", 192) 
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size,) * 3
        print(f"Patch size: {self.patch_size}")
        self.use_patch_bank_sampling = bool(self.params.get("USE_PATCH_BANK_SAMPLING", True))
        self.patch_banks_dir = self.params.get("PATCH_BANKS_DIR", str(Path(self.data_path) / "patch_banks"))
        self.sampler_pos_ratio = float(self.params.get("SAMPLER_POS_RATIO", 0.9))
        self.sampler_temperature = float(self.params.get("SAMPLER_TEMPERATURE", 0.7))
        self.sampler_fg_threshold = float(self.params.get("SAMPLER_FG_THRESHOLD", 1.72e-05))  # locked q25
        # Landmark-aware sampling knobs (optional)
        self.sampler_use_landmarks = bool(self.params.get("SAMPLER_USE_LANDMARKS", False))
        self.sampler_quota_bifurcation = float(self.params.get("SAMPLER_QUOTA_BIFURCATION", 0.0))
        self.sampler_quota_trachea = float(self.params.get("SAMPLER_QUOTA_TRACHEA", 0.0))
        # Initialize file lists
        self.train_files = []
        self.val_files = []
        self.off_val_files = []
        self.test_files = []

        # Initialize datasets and transforms to None to satisfy linters
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.off_val_ds = None
        self.deterministic_transforms = None
        self.random_transforms = None
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

    def get_data_paths(self):
        """Setup data file paths."""
        data_path_obj = Path(self.data_path)
        
        # Training and validation data (from npy_files)
        images_path = data_path_obj / "imagesTr"
        labels_path = data_path_obj / "labelsTr"
        lungs_path = data_path_obj / "lungsTr"
        
        if not images_path.is_dir() or not labels_path.is_dir():
            raise FileNotFoundError(f"Training data not found. Expected {images_path} and {labels_path}")
        
        # Get all training files
        all_files = []
        for img_file in sorted(images_path.glob('*.npy')):
            lbl_file = labels_path / img_file.name
            lung_file = lungs_path / img_file.name
            stem = img_file.stem
            meta_path = img_file.with_name(f"{stem}_meta.json")
            if lbl_file.exists() and lung_file.exists():
                all_files.append({
                    'image': str(img_file),
                    'meta' : str(meta_path) if meta_path.exists() else None,
                    'label': str(lbl_file),
                    'lung': str(lung_file)
                })
        
        if not all_files:
            raise ValueError("No valid training files found")
        
        val_size = max(1, int((1 - self.train_val_ratio) * len(all_files)))
        print(f"val_size value {val_size}")
        self.train_files = all_files[:-val_size]
        self.val_files = all_files[-val_size:]
        
        # Official validation data (from .npy files - now preprocessed consistently)
        dataset_root = Path(self.data_path).parent
        off_val_path = Path("./") / dataset_root / 'val'
        off_val_lung_path = Path("./") / dataset_root / 'val_lungs'
        # off_val_label_path = dataset_root / 'val_labels'  

        self.off_val_files = []
        # Check for validation labels and include them if available for metric testing.
        # self.has_val_labels = off_val_label_path.is_dir()
        
        if off_val_path.is_dir() and off_val_lung_path.is_dir():
            for img_file in sorted(off_val_path.glob('*.npy')):  # Changed from .nii.gz to .npy
                lung_file = off_val_lung_path / img_file.name
                stem = img_file.stem
                meta_path = img_file.with_name(f"{img_file.stem}_meta.json")
                file_dict = {
                    'image': str(img_file),
                    'meta': str(meta_path) if meta_path.exists() else None,
                    'lung': str(lung_file) if lung_file.exists() else None
                }
                
                # If validation labels exist, add them to the file dictionary.
                # if self.has_val_labels:
                #     label_file = off_val_label_path / img_file.name
                #     if label_file.exists():
                #         file_dict['label'] = str(label_file)
                
                # Only include if we have at least image and lung
                if file_dict['lung'] is not None:
                    self.off_val_files.append(file_dict)
        
        self.test_files = self.off_val_files  # Use off_val as test set
        
        print(f"Dataset split - Train: {len(self.train_files)}, Val: {len(self.val_files)}, "
              f"Off-val: {len(self.off_val_files)}, Test: {len(self.test_files)}")
        # print(f"Validation labels available: {self.has_val_labels}")
        # if self.has_val_labels:
        #     val_with_labels = sum(1 for f in self.off_val_files if 'label' in f)
        #     print(f"Validation files with labels: {val_with_labels}/{len(self.off_val_files)}")

    def get_deterministic_transforms_list(self, stage="train"):
        """
        Define the deterministic transforms as a list. The order is critical.
        """
        img_label_keys = ['image', 'lung', 'label'] if stage in ['train', 'val'] else ['image', 'lung']
        
        # Dynamically set the interpolation modes based on the keys
        if stage in ['train', 'val']:
            interpolation_modes = ("bilinear", "nearest", "nearest")
        else: # stage is 'test'
            interpolation_modes = ("bilinear", "nearest")

        transforms_list = [
            LoadImaged(
                keys=img_label_keys, reader=NumpyReader, ensure_channel_first=False,
                image_only=False, allow_missing_keys=True, meta_keys=[f"{k}_meta_dict" for k in img_label_keys]
            ),
        ]
        transforms_list.append(SetSpacingFromMeta(keys=img_label_keys, allow_missing_keys=True))
        # Add LoadOriginalNiftiHeaderd for test stage to preserve original metadata
        if stage == 'test':
            transforms_list.append(LoadOriginalNiftiHeaderd(keys=['image'], allow_missing_keys=True))
        
        transforms_list.extend([
            EnsureChannelFirstd(keys=img_label_keys, allow_missing_keys=True),
            EnsureTyped(keys=['image', 'lung', 'label'], track_meta=True, allow_missing_keys=True),
        ])
        # Ensure lung is binary and limited to {0,1} after load to avoid visualization artifacts
        transforms_list.append(
            BinarizeLungd(keys=["lung"], allow_missing_keys=True)
        )
        # IMPORTANT: apply Orientationd before sampling if using RandCrop, but AFTER sampling if using patch-bank
        if not self.use_patch_bank_sampling:
            print("Applying Orientationd before sampling, not using patch-bank")
            transforms_list.append(
                Orientationd(keys=img_label_keys, axcodes="RAS", allow_missing_keys=True)
            )
        
        # Intensity scaling - use pixel min/max values from params
        pixel_min = self.params.get("PIXEL_VALUE_MIN", -1000)
        pixel_max = self.params.get("PIXEL_VALUE_MAX", 600)
        norm_min = self.params.get("PIXEL_NORM_MIN", 0.0)
        norm_max = self.params.get("PIXEL_NORM_MAX", 1.0)
        
        transforms_list.append(
            ScaleIntensityRanged(
                keys=["image"], a_min=pixel_min, a_max=pixel_max, b_min=norm_min, b_max=norm_max, clip=True,
            )
        )

        if stage=="train":
            # If using patch-bank, avoid CropForegroundd here (we will extract exact ROI positions)
            if not self.use_patch_bank_sampling:
                print("Applying CropForegroundd, not using patch-bank")
                transforms_list.append(
                    CropForegroundd(keys=img_label_keys, source_key="lung", allow_missing_keys=True, margin=[1,1,1])
                )
        elif stage=="val":
            transforms_list.append(
                CropForegroundd(keys=img_label_keys, source_key="lung", allow_missing_keys=True, margin=[1,1,25])
            )
        else:
            transforms_list.append(
                CropForegroundd(keys=img_label_keys, source_key="lung", allow_smaller = True, margin=[1,1,50])
            )
        
        
        # dtype choice
        if stage == "train" and self.compute_dtype == torch.float16:
            image_dtype = torch.float16
        else:
            image_dtype = torch.float32
        
        return transforms_list

    def get_random_transforms(self, stage="train"):
        img_label_keys = ["image", "lung", "label"]
        img_keys = ["image"]
        
        transforms = []
        print(f"use_patch_bank_sampling: {self.use_patch_bank_sampling}")
        print(f"patch_banks_dir: {self.patch_banks_dir}")
        print(f"sampler_pos_ratio: {self.sampler_pos_ratio}")
        print(f"sampler_temperature: {self.sampler_temperature}")
        print(f"sampler_fg_threshold: {self.sampler_fg_threshold}")
        if stage == "train":
            # If using patch-bank sampling, do not add SpatialPadd here; we will crop exact patches instead in patch-bank sampler
            if not self.use_patch_bank_sampling:
                transforms.append(
                    SpatialPadd(
                    keys=img_label_keys,
                    spatial_size=self.patch_size,
                    method="symmetric",
                    mode="constant",
                    allow_missing_keys=True
                    )
                )
            # Sampler selection
            if self.use_patch_bank_sampling and SamplePatchFromBankd is not None:
                print("Applying SamplePatchFromBankd, using patch-bank")
                transforms.append(
                    SamplePatchFromBankd(
                        keys=img_label_keys,
                        patch_banks_dir=self.patch_banks_dir,
                        spatial_size=tuple(int(v) for v in self.patch_size),
                        num_samples=1,
                        pos_ratio=self.sampler_pos_ratio,
                        temperature=self.sampler_temperature,
                        fg_threshold=self.sampler_fg_threshold,
                        use_landmarks=self.sampler_use_landmarks,
                        quota_bifurcation=self.sampler_quota_bifurcation,
                        quota_trachea=self.sampler_quota_trachea,
                        allow_missing_keys=True,
                    )
                )
                # Orientation after sampling if using patch-bank, to keep ROI indices consistent with audited frame
                transforms.append(
                    Orientationd(keys=img_label_keys, axcodes="RAS", allow_missing_keys=True)
                )
            else:
                print("Applying RandCropByPosNegLabeld, not using patch-bank")
                transforms.append(
                        RandCropByPosNegLabeld(
                            keys=img_label_keys,
                            label_key="label",
                            spatial_size=self.patch_size,
                            pos=8, neg=1, num_samples=1,
                            allow_missing_keys=True,
                            allow_smaller=True
                        )
                )
            
        elif stage == "val":
            transforms.extend([
                SpatialPadd(
                    keys=img_label_keys,
                    spatial_size=self.patch_size,
                    method="symmetric",
                    mode="constant",
                    allow_missing_keys=True
                )
            ])
        if stage == "train" and self.do_augment:
            transforms.extend([
                RandFlipd(keys=img_label_keys, prob=0.5, spatial_axis=[0, 1, 2], allow_missing_keys=True),
                RandRotate90d(keys=img_label_keys, prob=0.5, max_k=3, allow_missing_keys=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), # Randomly scale intensity by +/- 10%
            ])
        #downcast for mixed precision
        if stage == "train" and self.compute_dtype == torch.float16:
            transforms.append(CastToTyped(keys=["image"], dtype=torch.float16, allow_missing_keys=True))
        return Compose(transforms)

    def get_val_transforms(self):
        """Gets complete transforms for validation (deterministic + center crop)."""
        # Flatten the transforms from nested Compose objects into a single list
        deterministic_list = self.get_deterministic_transforms_list(stage="val")
        val_crop_list = self.get_random_transforms(stage="val").transforms
        return Compose(deterministic_list + list(val_crop_list))

    def setup(self, stage=None):
        """Setup datasets and transforms with a robust caching strategy."""
        if self._setup_called:
            print("Data module already set up. Skipping.")
            return

        print(f"Setting up data module for stage: {stage}...")
        self.get_data_paths()
        self.deterministic_transforms_list = self.get_deterministic_transforms_list(stage="train")
        self.train_transforms = Compose(self.deterministic_transforms_list + list(self.get_random_transforms(stage="train").transforms))
        
        self.val_transforms = self.get_val_transforms()
        self.test_transforms_list = self.get_deterministic_transforms_list(stage="test")
        self.test_transforms = Compose(self.test_transforms_list)

        if stage == 'fit' or stage is None:
            # The deterministic transforms for caching must be a Compose object
            deterministic_compose = Compose(self.deterministic_transforms_list)
            random_compose = self.get_random_transforms(stage="train") 

            if self.cache_dataset: # Fallback to CacheDataset (in-RAM)
                print("Using CacheDataset to cache deterministic transforms...")
                num_cache_workers = max(1, int(self.num_workers * 0.75))
                _cached_deterministic_ds = CacheDataset(
                    data=self.train_files,
                    transform=deterministic_compose,
                    cache_rate=self.cache_rate,
                    num_workers=num_cache_workers
                )
                self.train_ds = Dataset(data=_cached_deterministic_ds, transform=random_compose)
                print(f"  - CacheDataset configured. Random transforms will be applied on-the-fly.")
            elif self.use_persistent_dataset:
                print("Using PersistentDataset (disk-backed cache) for deterministic transforms...")
                _persistent_train = PersistentDataset(
                    data=self.train_files,
                    transform=deterministic_compose,
                    cache_dir=str(self.cache_dir / "train")
                )
                self.train_ds = Dataset(data=_persistent_train, transform=random_compose)
                print(f"  - PersistentDataset configured at: {self.cache_dir / 'train'}")
            else:
                print("Using regular Dataset (no caching).")
                self.train_ds = Dataset(data=self.train_files, transform=self.train_transforms)
                
        if self.use_persistent_dataset:
            self.val_ds = PersistentDataset(
                data=self.val_files,
                transform=self.val_transforms,
                cache_dir=str(self.cache_dir / "val")
            )
            self.off_val_ds = PersistentDataset(
                data=self.off_val_files,
                transform=self.test_transforms,
                cache_dir=str(self.cache_dir / "off_val")
            )
            self.test_ds = PersistentDataset(
                data=self.test_files,
                transform=self.test_transforms,
                cache_dir=str(self.cache_dir / "test")
            )
        else:
            self.val_ds = Dataset(data=self.val_files, transform=self.val_transforms)
            self.off_val_ds = Dataset(data=self.off_val_files, transform=self.test_transforms)
            self.test_ds = Dataset(data=self.test_files, transform=self.test_transforms)

        print("Data module setup complete.")
        if hasattr(self, 'train_ds') and self.train_ds is not None: print(f"Training dataset size: {len(self.train_ds)}")
        if hasattr(self, 'val_ds') and self.val_ds is not None: print(f"Validation dataset size: {len(self.val_ds)}")
        self._setup_called = True     
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _create_dataloader(self, dataset, batch_size, shuffle=False, num_workers=None):
        """
        Create optimized dataloader with intelligent worker configuration.
        
        Args:
            dataset: The dataset to create dataloader for
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the data
            num_workers: Override for number of workers (auto-calculated if None)
        """
        if not dataset or (hasattr(dataset, 'data') and not dataset.data):
            print(f"Warning: Empty dataset provided to dataloader")
            return DataLoader(Dataset([]), batch_size=batch_size)
        
        prefetch_factor=None
        use_persistent = False
        num_workers = 0 if num_workers==None else num_workers
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=use_persistent,
            prefetch_factor=prefetch_factor,
            collate_fn=pad_list_data_collate,
            worker_init_fn=worker_init_fn_enhanced if num_workers > 0 else None,
            drop_last=shuffle,  # Drop last incomplete batch for training
        )

    def _create_test_dataloader(self, dataset, batch_size, shuffle=False):
        """Create test dataloader with memory optimizations."""
        if not dataset or not dataset.data:
            print(f"Warning: Empty dataset provided to test dataloader")
            return DataLoader(Dataset([]), batch_size=batch_size)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False,  # Disable pin_memory
            persistent_workers=False,
            prefetch_factor=None,  # Disable prefetch
            collate_fn=pad_list_data_collate,
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._create_dataloader(self.train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._create_dataloader(self.val_ds, self.test_batch_size, shuffle=False, num_workers=1)

    def aff_training_val_dataloader(self):
        """Creates a dataloader for the validation split of the training data, used for testing."""
        if not hasattr(self, 'val_ds') or self.val_ds is None:
            self.setup(stage='fit') 
        return self._create_dataloader(self.val_ds, self.test_batch_size, shuffle=False, num_workers=0)

    def off_val_dataloader(self):
        """Create official validation dataloader."""
        return self._create_test_dataloader(self.off_val_ds, self.test_batch_size, shuffle=False)

    def test_dataloader(self):
        """
        Returns the appropriate dataloader for the test phase
        """
        # Allow choosing which dataset to iterate
        chosen = (self.args.test_dataset if self.args and hasattr(self.args, 'test_dataset') else 'off_val')
        if self.args and self.args.metric_testing:
            if chosen == 'val':
                print("Metric testing enabled: using validation split (with labels) for the 'test' stage.")
                return self.aff_training_val_dataloader()
            else:
                print("Metric testing enabled: using official validation set (off_val) for the 'test' stage.")
                return self._create_test_dataloader(self.off_val_ds, self.test_batch_size, shuffle=False)
        else:
            if chosen == 'val':
                print("Inference mode: using validation split (with labels) for the 'test' stage.")
                return self.aff_training_val_dataloader()
            else:
                print("Inference mode: using official validation set (off_val) for the 'test' stage.")
                return self._create_test_dataloader(self.test_ds, self.test_batch_size, shuffle=False)
