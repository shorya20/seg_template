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
import os
class SaveImageWithOriginalMetadatad(MapTransform):
    """
    Save predictions like MONAI's SaveImaged, but first resample the MetaTensor
    from the processed grid back to the original voxel grid and save using the
    original 4x4 affine. 
    """
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        output_dir: Union[str, Path],
        output_postfix: str = "seg",
        separate_folder: bool = True,
        print_log: bool = True,
        allow_missing_keys: bool = False,
        resample_full_volume: bool = True,
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
        thr = float(os.getenv("PRED_THRESHOLD", "0.5"))

        for key in self.key_iterator(d):
            pred_tensor = d.get(key)
            if pred_tensor is None:
                continue
            
            if isinstance(pred_tensor, MetaTensor) and hasattr(pred_tensor, "meta"):
                meta = pred_tensor.meta
            else:
                meta = d.get(f"{key}_meta_dict", {}) or d.get("image_meta_dict", {})
                if not meta:
                    print(f"Warning: {key} has no metadata available, skipping")
                    continue

            filename = meta.get("filename_or_obj", "unknown")
            if isinstance(filename, (list, tuple)) and len(filename) > 0:
                filename = filename[0]

            meta_path = Path(str(filename).replace(".npy", "_meta.json"))
            if not meta_path.exists():
                print(f"Warning: Metadata file not found: {meta_path}")
                continue

            with open(meta_path, "r") as f:
                json_meta = json.load(f)

            proc_spacing   = np.array(json_meta.get("processed_spacing"))
            proc_direction = np.array(json_meta.get("processed_direction")).reshape(3, 3)
            proc_origin    = np.array(json_meta.get("processed_origin"))

            # Convert RAS -> LPS for SimpleITK output frame 
            orig_spacing      = np.array(json_meta.get("original_spacing"))
            _orig_dir_ras     = np.array(json_meta.get("original_direction")).reshape(3, 3)
            _orig_origin_ras  = np.array(json_meta.get("original_origin"))
            ras2lps           = np.diag([-1.0, -1.0, 1.0])
            orig_direction    = ras2lps @ _orig_dir_ras
            orig_origin       = (ras2lps @ _orig_origin_ras.reshape(3, 1)).ravel()
            orig_shape_xyz    = list(json_meta.get("original_spatial_shape"))  # [X, Y, Z]
            orig_affine       = np.array(json_meta.get("original_affine"))

            # Convert to numpy [Z, Y, X] (as said in the original report)
            pred_np = pred_tensor.cpu().numpy()
            if pred_np.ndim == 4 and pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)

            if pred_np.dtype not in (np.uint8, np.int16, np.int32):
                pred_np = (pred_np > float(thr)).astype(np.uint8)

            nz = np.where(pred_np > 0)
            if len(nz[0]) == 0:
                z_min = z_max = y_min = y_max = x_min = x_max = None
            else:
                z_min, z_max = nz[0].min(), nz[0].max()
                y_min, y_max = nz[1].min(), nz[1].max()
                x_min, x_max = nz[2].min(), nz[2].max()

            if len(nz[0]) == 0:
                print(f"Warning: Empty prediction for {filename}")
                out_xyz = np.transpose(np.zeros(orig_shape_xyz[::-1], dtype=np.uint8), (2, 1, 0))
            elif self.resample_full_volume:
                # Full volume resampling (robust)
                sitk_full = sitk.GetImageFromArray(pred_np.astype(np.uint8))
                sitk_full.SetSpacing(tuple(proc_spacing))
                sitk_full.SetDirection(proc_direction.flatten().tolist())
                sitk_full.SetOrigin(tuple(proc_origin))

                resampler = sitk.ResampleImageFilter()
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetOutputSpacing(tuple(orig_spacing))
                resampler.SetOutputDirection(orig_direction.flatten().tolist())
                resampler.SetOutputOrigin(tuple(orig_origin))
                resampler.SetSize([int(s) for s in orig_shape_xyz])
                resampler.SetDefaultPixelValue(0)

                resampled = resampler.Execute(sitk_full)
                out_xyz = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))

                if self.print_log:
                    print(f"  Full volume resampling: {int(np.sum(out_xyz > 0)):,} non-zero voxels")
            else:
                pred_bbox = pred_np[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
                idx_xyz = np.array([x_min, y_min, z_min], dtype=np.float64)
                vox_xyz = np.array(proc_spacing, dtype=np.float64)
                bbox_origin_xyz = proc_origin + proc_direction @ (idx_xyz * vox_xyz)

                sitk_bbox = sitk.GetImageFromArray(pred_bbox.astype(np.uint8))
                sitk_bbox.SetSpacing(tuple(proc_spacing))
                sitk_bbox.SetDirection(proc_direction.flatten().tolist())
                sitk_bbox.SetOrigin(tuple(bbox_origin_xyz))

                resampler = sitk.ResampleImageFilter()
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetOutputSpacing(tuple(orig_spacing))
                resampler.SetOutputDirection(orig_direction.flatten().tolist())
                resampler.SetOutputOrigin(tuple(orig_origin))
                resampler.SetSize([int(s) for s in orig_shape_xyz])
                resampler.SetTransform(sitk.Transform())

                resampled = resampler.Execute(sitk_bbox)
                out_xyz = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))

            base_name = Path(filename).stem
            if base_name.endswith("_0000"):
                base_name = base_name[:-5]
            out_name = f"{base_name}_{self.output_postfix}.nii.gz"

            if self.separate_folder:
                sub = self.output_dir / base_name
                sub.mkdir(parents=True, exist_ok=True)
                out_path = sub / out_name
            else:
                out_path = self.output_dir / out_name

            nii = nib.Nifti1Image(out_xyz.astype(np.uint8), orig_affine)
            hdr = nii.header
            hdr.set_data_dtype(np.uint8)
            try:
                hdr.set_qform(orig_affine, code=1)
                hdr.set_sform(orig_affine, code=1)
                hdr.set_xyzt_units("mm", "sec")
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
