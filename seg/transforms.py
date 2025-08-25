"""
Custom transforms for handling prediction resampling back to original NIfTI space.
"""
import numpy as np
import torch
from monai.transforms.transform import MapTransform
from monai.data.meta_tensor import MetaTensor
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path
import json
import SimpleITK as sitk
import os
from monai.transforms import MapTransform, Transform
from monai.config import DtypeLike
from monai.utils import convert_to_numpy, convert_to_tensor


def _spacing_direction_origin_from_affine(aff):
    # Ensure aff is a NumPy array before processing
    if isinstance(aff, torch.Tensor):
        aff = aff.cpu().numpy()
    
    RZS = aff[:3, :3]
    spacing = np.linalg.norm(RZS, axis=0)
    direction = (RZS / spacing).reshape(-1).tolist()  # row-major
    origin = aff[:3, 3].tolist()
    return tuple(spacing.tolist()), tuple(direction), tuple(origin)

class DebugMetadatad(MapTransform):
    """
    Debug transform to print metadata at various stages of the pipeline.
    """
    def __init__(self, keys=["image"], prefix="", allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)
        self.prefix = prefix
    def __call__(self, data):
        d = dict(data)
        print(f"\n{'='*60}")
        print(f"[DebugMetadatad] {self.prefix}")
        print(f"{'='*60}")
        if "image_meta_dict" in d:
            meta = d["image_meta_dict"]
            print("Image metadata keys:", list(meta.keys()) if isinstance(meta, dict) else "Not a dict")
        
            if isinstance(meta, dict):
                for key in ["spatial_shape", "original_spatial_shape", "original_affine", "affine", "filename_or_obj"]:
                    if key in meta:
                        value = meta[key]
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            print(f"  {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}, dtype={value.dtype}")
                            if key == "original_spatial_shape":
                                print(f"    Value: {value}")
                        else:
                            print(f"  {key}: {value}")
        
        for key in self.key_iterator(d):
            item = d.get(key)
            if item is not None:
                print(f"\n{key} tensor info:")
                if isinstance(item, MetaTensor):
                    print(f"  Shape: {item.shape}")
                    print(f"  Has meta: {hasattr(item, 'meta')}")
                    if hasattr(item, 'meta') and item.meta:
                        print(f"  Meta keys: {list(item.meta.keys())}")
                        if 'original_affine' in item.meta:
                            print(f"  Original affine available: True")
                        if 'original_spatial_shape' in item.meta:
                            print(f"  Original shape: {item.meta['original_spatial_shape']}")
                elif isinstance(item, torch.Tensor):
                    print(f"  Shape: {item.shape} (regular tensor)")
                elif isinstance(item, np.ndarray):
                    print(f"  Shape: {item.shape} (numpy array)")
        
        print(f"{'='*60}\n")
        
        return d

