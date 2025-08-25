from monai.transforms.transform import MapTransform
import json
from scipy.ndimage import binary_erosion, label as nd_label
from scipy.ndimage import distance_transform_edt
from typing import Union, Sequence
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from monai.data import MetaTensor


class SaveImageWithOriginalMetadatad(MapTransform):
    """
    Save predictions like MONAI's SaveImaged, but first resample the MetaTensor
    from the processed grid (recorded in the *_meta.json produced by convert_to_numpy.py)
    back to the ORIGINAL voxel grid (spacing, direction, origin, shape), then save using
    the original 4x4 affine. Uses nearest-neighbor for labels.

    Args:
        keys: key or list of keys to save
        output_dir: target directory
        output_postfix: filename postfix (default: "seg")
        separate_folder: if True, save under a subfolder per case
        print_log: verbose prints
    """
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        output_dir: Union[str, Path],
        output_postfix: str = "seg",
        separate_folder: bool = True,
        print_log: bool = True,
        allow_missing_keys: bool = False,
        resample_full_volume: bool = True,  # New option to resample full volume instead of bbox
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.output_dir = Path(output_dir)
        self.output_postfix = output_postfix
        self.separate_folder = separate_folder
        self.print_log = print_log
        self.resample_full_volume = resample_full_volume
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            pred_tensor = d.get(key)
            if pred_tensor is None:
                continue

            # Get metadata from the prediction tensor
            if not isinstance(pred_tensor, MetaTensor) or not hasattr(pred_tensor, 'meta'):
                print(f"Warning: {key} is not a MetaTensor with metadata, skipping")
                continue

            meta = pred_tensor.meta
            filename = meta.get('filename_or_obj', 'unknown')
            
            # Read JSON metadata for coordinate system info
            meta_path = Path(str(filename).replace('.npy', '_meta.json'))
            if not meta_path.exists():
                print(f"Warning: Metadata file not found: {meta_path}")
                continue
                
            with open(meta_path, 'r') as f:
                json_meta = json.load(f)

            # Extract coordinate system info
            proc_spacing = np.array(json_meta.get('processed_spacing'))
            proc_direction = np.array(json_meta.get('processed_direction')).reshape(3, 3)
            proc_origin = np.array(json_meta.get('processed_origin'))
            
            orig_spacing = np.array(json_meta.get('original_spacing'))
            orig_direction = np.array(json_meta.get('original_direction')).reshape(3, 3)
            orig_origin = np.array(json_meta.get('original_origin'))
            orig_shape_xyz = list(json_meta.get('original_spatial_shape'))  # [X, Y, Z]
            orig_affine = np.array(json_meta.get('original_affine'))

            # Convert MetaTensor -> numpy, squeeze [C, Z, Y, X] -> [Z, Y, X]
            pred_np = pred_tensor.cpu().numpy()
            if pred_np.ndim == 4 and pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)
            # Ensure binary prediction
            if pred_np.dtype not in (np.uint8, np.int16, np.int32):
                pred_np = (pred_np > 0.5).astype(np.uint8)

            # Resample prediction to original space
            nonzero_coords = np.where(pred_np > 0)
            # Precompute bbox indices if any non-zero voxels exist to avoid undefined locals later
            if len(nonzero_coords[0]) == 0:
                z_min = z_max = y_min = y_max = x_min = x_max = None
            else:
                z_min, z_max = nonzero_coords[0].min(), nonzero_coords[0].max()
                y_min, y_max = nonzero_coords[1].min(), nonzero_coords[1].max()
                x_min, x_max = nonzero_coords[2].min(), nonzero_coords[2].max()

            if len(nonzero_coords[0]) == 0:
                print(f"Warning: Empty prediction for {filename}")
                # Create empty output in original space
                empty_pred = np.zeros(orig_shape_xyz[::-1], dtype=np.uint8)  # ZYX order
                out_xyz = np.transpose(empty_pred, (2, 1, 0))
            elif self.resample_full_volume:
                # Resample the full volume (more robust, avoids bbox coordinate issues)
                sitk_full = sitk.GetImageFromArray(pred_np.astype(np.uint8))
                sitk_full.SetSpacing(tuple(proc_spacing))
                sitk_full.SetDirection(proc_direction.flatten().tolist())
                sitk_full.SetOrigin(tuple(proc_origin))

                # Resample full volume to original coordinate system
                resampler = sitk.ResampleImageFilter()
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetOutputSpacing(tuple(orig_spacing))
                resampler.SetOutputDirection(orig_direction.flatten().tolist())
                resampler.SetOutputOrigin(tuple(orig_origin))
                resampler.SetSize([int(s) for s in orig_shape_xyz])
                resampler.SetDefaultPixelValue(0)

                resampled = resampler.Execute(sitk_full)
                resampled_zyx = sitk.GetArrayFromImage(resampled)  # (Z, Y, X)
                out_xyz = np.transpose(resampled_zyx, (2, 1, 0))

                # Print non-zero voxel info
                nonzero_count = np.sum(out_xyz > 0)
                if self.print_log:
                    print(f"  Full volume resampling: {nonzero_count:,} non-zero voxels")
            else:
                # Original bbox-based resampling (kept for backward compatibility)
                # Extract only the bounding box region
                pred_bbox = pred_np[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                
                # Calculate physical coordinates of bounding box corners in processed space
                # SimpleITK uses (X, Y, Z) indexing, so convert ZYX -> XYZ
                index_xyz = np.array([x_min, y_min, z_min], dtype=np.float64)
                voxel_size_xyz = np.array(proc_spacing, dtype=np.float64)
                # physical = origin + direction @ (index * spacing)
                bbox_origin_xyz = proc_origin + proc_direction @ (index_xyz * voxel_size_xyz)
                bbox_size_xyz = np.array([x_max-x_min+1, y_max-y_min+1, z_max-z_min+1])
                
                # Build SimpleITK image from bounding box only
                sitk_bbox = sitk.GetImageFromArray(pred_bbox.astype(np.uint8))
                sitk_bbox.SetSpacing(tuple(proc_spacing))
                sitk_bbox.SetDirection(proc_direction.flatten().tolist())
                sitk_bbox.SetOrigin(tuple(bbox_origin_xyz))

                # Resample bounding box to original coordinate system
                resampler = sitk.ResampleImageFilter()
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetOutputSpacing(tuple(orig_spacing))
                resampler.SetOutputDirection(orig_direction.flatten().tolist())
                resampler.SetOutputOrigin(tuple(orig_origin))
                resampler.SetSize([int(s) for s in orig_shape_xyz])
                resampler.SetTransform(sitk.Transform())

                resampled = resampler.Execute(sitk_bbox)
                resampled_zyx = sitk.GetArrayFromImage(resampled)  # (Z, Y, X)
                out_xyz = np.transpose(resampled_zyx, (2, 1, 0))

            # Create output path
            base_name = Path(filename).stem  # e.g., ATM_281_0000
            if base_name.endswith('_0000'):
                base_name = base_name[:-5]
            out_name = f"{base_name}_{self.output_postfix}.nii.gz"
            if self.separate_folder:
                sub = self.output_dir / base_name
                sub.mkdir(parents=True, exist_ok=True)
                out_path = sub / out_name
            else:
                out_path = self.output_dir / out_name

            # Write with ORIGINAL affine; set header codes/units
            nii = nib.Nifti1Image(out_xyz.astype(np.uint8), orig_affine)
            hdr = nii.header
            hdr.set_data_dtype(np.uint8)
            try:
                hdr.set_qform(orig_affine, code=1)
                hdr.set_sform(orig_affine, code=1)
                hdr.set_xyzt_units('mm', 'sec')
            except Exception:
                pass

            nib.save(nii, str(out_path))

            if self.print_log:
                print(f"Saved (resampled) to: {out_path}")
                print(f"  Out shape (X,Y,Z): {out_xyz.shape}")
                if z_min is not None:
                    print(f"  Bbox in processed space: Z[{z_min}:{z_max+1}], Y[{y_min}:{y_max+1}], X[{x_min}:{x_max+1}]")
                    print(f"  Non-zero voxels: {np.sum(out_xyz > 0):,}")

            d[f"{key}_saved_path"] = str(out_path)
        return d

