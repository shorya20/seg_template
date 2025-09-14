from __future__ import annotations
import os
import sys
import json
import gc
import argparse
import logging
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional
from joblib import Parallel, delayed
import nibabel as nib

from skimage.measure import regionprops, label as skimage_label
from skimage.segmentation import clear_border
from skimage.morphology import disk
from skimage.filters import threshold_otsu

from scipy.ndimage import (
    binary_fill_holes, binary_opening, binary_closing, binary_erosion,
    label, generate_binary_structure, binary_dilation, distance_transform_edt
)

import traceback
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import SimpleITK as sitk
    from lungmask import mask as lungmask_model
except ImportError:
    print("SimpleITK or lungmask is not installed. Please install it using 'pip install SimpleITK-SimpleElastix lungmask'.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def _save_meta(base_path: str, meta: dict) -> None:
    for k, v in meta.items():
        if isinstance(v, tuple):
            meta[k] = list(v)
    with open(base_path + "_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)


def _clear_border_xy_per_slice(seg3d: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg3d, dtype=bool)
    for z in range(seg3d.shape[0]):
        out[z] = clear_border(seg3d[z].astype(bool))
    return out

def _air_body_candidates(image_for_masking: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust, model-free lung candidate from intensities on the SAME grid used for lungmask.
    Steps:
    1) Build a coarse thoracic body mask (percentile-based) and fill holes.
    2) Map intensities to a pseudo-HU using robust percentiles (5–95) *inside body*.
    3) Inside the coarse body, compute an Otsu threshold to separate air-like from soft tissue.
    4) Cleanup per-slice clear_border; keep the 2 largest 3D components; fill holes.
    5) Fallback: looser percentile rule if result is implausibly small.
    Returns:
        cand: uint8 {0,1} candidate lung (air ∧ body)
        body: uint8 {0,1} coarse body mask
    """
    np_img = sitk.GetArrayFromImage(image_for_masking).astype(np.float32)
    p10, p50 = np.percentile(np_img, [10, 50])
    body_thr = p10 + 0.05 * (p50 - p10)
    body0 = np_img > body_thr

    lab, n = skimage_label(body0.astype(np.uint8), return_num=True)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        keep = np.argmax(sizes[1:]) + 1 if sizes.size > 1 else 0
        body0 = (lab == keep)

    body0 = binary_opening(body0, structure=np.ones((3, 3, 3)), iterations=1)
    body0 = binary_closing(body0, structure=np.ones((3, 3, 3)), iterations=1)
    body0 = binary_fill_holes(body0)

    if not np.any(body0):
        thr35 = np.percentile(np_img, 35)  
        body0 = np_img > thr35
        if body0.any():
            lab, n = skimage_label(body0.astype(np.uint8), return_num=True)
            if n > 1:
                sizes = np.bincount(lab.ravel()); body0 = (lab == (np.argmax(sizes[1:]) + 1))
            body0 = binary_fill_holes(body0)
        else:
            empty = np.zeros_like(np_img, dtype=np.uint8)
            return empty, empty  # (cand, body) both empty; caller will handle

    vals = np_img[body0 > 0]
    p05, p95 = np.percentile(vals, [5, 95]) if vals.size > 1000 else (np.min(vals), np.max(vals))
    if p95 - p05 < 1e-3:
        scale, offset = 1.0, 0.0
    else:
        scale = 2000.0 / float(p95 - p05)
        offset = -1000.0 - scale * float(p05)
    pseudo_hu = np_img * scale + offset

    try:
        t = float(threshold_otsu(pseudo_hu[body0 > 0]))
    except Exception:
        t = -400.0
    t = np.clip(t, -800.0, -200.0) 
    cand = (pseudo_hu <= t) & body0

    cand = _clear_border_xy_per_slice(cand)
    lab, n = skimage_label(cand.astype(np.uint8), return_num=True)
    if n > 2:
        sizes = np.bincount(lab.ravel())
        order = np.argsort(sizes[1:])[::-1] + 1
        cand = np.isin(lab, order[:2])
    cand = binary_fill_holes(cand).astype(np.uint8)
    frac = float(cand.sum()) / float(np_img.size)
    if frac < 0.001:
        body_vals = np_img[body0 > 0]
        if body_vals.size > 0:
            loose_thr = np.percentile(body_vals, 20)
            cand2 = (np_img <= loose_thr) & body0
            cand2 = _clear_border_xy_per_slice(cand2)
            lab, n = skimage_label(cand2.astype(np.uint8), return_num=True)
            if n > 2:
                sizes = np.bincount(lab.ravel())
                order = np.argsort(sizes[1:])[::-1] + 1
                cand2 = np.isin(lab, order[:2])
            cand2 = binary_fill_holes(cand2).astype(np.uint8)
            if cand2.sum() > cand.sum():
                cand = cand2

    return cand.astype(np.uint8), body0.astype(np.uint8)


def _load_and_preprocess_sitk(nifti_path: str, is_label: bool = False) -> sitk.Image | None:
    """
    Load a NIfTI with nibabel, build a SimpleITK image on the original grid (LPS),
    and fix intensities so air is HU-like. Labels are passed through (nearest-neighbor later).
    """
    try:
        import numpy as _np
        import nibabel as _nib
        from scipy.ndimage import binary_closing as _bin_close, binary_fill_holes as _bin_fill

        # --- 1) Load with nibabel and pull float array with header scaling (if any) ---
        nii = _nib.load(nifti_path)
        data = nii.get_fdata(dtype=_np.float32, caching='unchanged') 
        on_disk_kind = _np.dtype(nii.get_data_dtype()).kind  # 'i','u','f'

        # --- 2) Build LPS meta from the affine (nibabel uses RAS) ---
        aff = nii.affine
        spacing_xyz = _np.sqrt(_np.sum(aff[:3, :3] ** 2, axis=0))  # col norms
        R_ras = aff[:3, :3] / _np.maximum(spacing_xyz, 1e-8)
        t_ras = aff[:3, 3]
        ras2lps = _np.diag([-1.0, -1.0, 1.0])
        R_lps = ras2lps @ R_ras
        t_lps = (ras2lps @ t_ras.reshape(3, 1)).ravel()
        arr_zyx = _np.transpose(data, (2, 1, 0)).astype(_np.float32, copy=False)

        if is_label:
            img = sitk.GetImageFromArray(arr_zyx.astype(_np.uint8, copy=False))
            img.SetSpacing(tuple(map(float, spacing_xyz.tolist())))
            img.SetDirection(R_lps.flatten().tolist())
            img.SetOrigin(t_lps.tolist())
            return img

        integer_like = (on_disk_kind in ('i', 'u')) and (_np.nanmin(arr_zyx) >= 0.0)
        if integer_like:
            x = arr_zyx - 1024.0  # put air near −1000 HU (water ~0) for lungmask robustness
        else:
            x = arr_zyx
            p10, p50 = _np.nanpercentile(x, [10, 50])
            thr = p10 + 0.05 * (p50 - p10)
            body = x > thr
            # keep largest 3D component
            try:
                from scipy.ndimage import label as _ndlabel
                lab, n = _ndlabel(body.astype(_np.uint8))
                if n > 1:
                    sizes = _np.bincount(lab.ravel())
                    keep = _np.argmax(sizes[1:]) + 1
                    body = (lab == keep)
            except Exception:
                pass
            body = _bin_fill(_bin_close(body, structure=_np.ones((3, 3, 3), _np.uint8)))
            vals = x[body]
            if vals.size > 1000:
                hist, edges = _np.histogram(vals, bins=512)
                centers = 0.5 * (edges[:-1] + edges[1:])
                air_peak = float(centers[_np.argmax(hist)])
                if not (-1100.0 <= air_peak <= -800.0):
                    x = x + (-1000.0 - air_peak)  
        x = _np.nan_to_num(x, copy=False).astype(_np.float32, copy=False)
        img = sitk.GetImageFromArray(x)
        img.SetSpacing(tuple(map(float, spacing_xyz.tolist())))
        img.SetDirection(R_lps.flatten().tolist())
        img.SetOrigin(t_lps.tolist())
        return img

    except Exception as e:
        logger.error(f"Error during nibabel/SITK loading/preprocessing for {nifti_path}: {e}")
        return None



def safe_convert_nifti_file(file, output_dir, is_label=False, target_spacing_xyz=None, antialias=True, force: bool=False):
    """
    Safely convert and resample a NIfTI file to NumPy format, writing a JSON metadata sidecar.
    Respects `force` to overwrite existing .npy.
    """
    try:
        # Always read metadata from nibabel for consistency
        nii_orig = nib.load(file)
        original_affine = nii_orig.affine
        original_shape_nii = nii_orig.shape
        original_spacing_from_affine = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
        original_direction = (original_affine[:3, :3] / np.maximum(original_spacing_from_affine, 1e-6)).tolist()
        original_origin = original_affine[:3, 3].tolist()

        fn = os.path.basename(file)
        stem = os.path.splitext(os.path.splitext(fn)[0])[0]
        out_npy = os.path.join(output_dir, f"{stem}.npy")
        if os.path.exists(out_npy) and not force:
            logger.debug("%s already exists – skipping", out_npy)
            return True, None

        img_sitk = _load_and_preprocess_sitk(file, is_label=is_label)
        if img_sitk is None:
            return False, f"Failed to load or preprocess {file}"
        try:
            oriented = sitk.DICOMOrient(img_sitk, "LPS")
        except Exception:
            oriented = img_sitk
            
        if target_spacing_xyz and len(target_spacing_xyz) == 3:
            target_spacing = tuple(float(s) for s in target_spacing_xyz)
            o_size = oriented.GetSize()
            o_spacing = oriented.GetSpacing()
            new_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(o_size, o_spacing, target_spacing)]
            R = sitk.ResampleImageFilter()
            R.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
            R.SetDefaultPixelValue(0.0 if is_label else -1024.0)
            R.SetOutputSpacing(target_spacing)
            R.SetSize(new_size)
            R.SetOutputDirection(oriented.GetDirection())
            R.SetOutputOrigin(oriented.GetOrigin())
            R.SetTransform(sitk.Transform())
            img_proc = R.Execute(oriented)
        else:
            img_proc = oriented

        final = sitk.GetArrayFromImage(img_proc)
        if final.size == 0:
            return False, f"Empty array after resampling for {file}"

        final = final.astype(np.uint8 if is_label else np.float32)
        np.save(out_npy, final)

        meta = {
            'original_filename': fn,
            'original_affine': original_affine.tolist(),
            'original_spatial_shape': list(original_shape_nii),
            'original_spacing': original_spacing_from_affine.tolist(),
            'original_size': list(original_shape_nii),
            'original_direction': original_direction,
            'original_origin': original_origin,
            'processed_spacing': list(img_proc.GetSpacing()),
            'processed_size': list(img_proc.GetSize()),
            'processed_direction': list(img_proc.GetDirection()),
            'processed_origin': list(img_proc.GetOrigin()),
            'numpy_array_shape_zyx': list(final.shape),
            'is_label': is_label,
            'target_spacing_if_resampled': target_spacing_xyz if target_spacing_xyz else "N/A",
        }
        base = os.path.splitext(os.path.basename(out_npy))[0]
        _save_meta(os.path.join(output_dir, base), meta)
        return True, None

    except Exception as e:
        return False, f"Error during SimpleITK processing for {file}: {str(e)}\n{traceback.format_exc()}"


def convert_nifti_to_numpy(nifti_path, output_dir, file_pattern="*.nii.gz",
                        is_label=False, target_spacing_xyz=None, num_workers=6, force: bool=False):
    """Parallel NIfTI→NumPy conversion with metadata writing."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(nifti_path, file_pattern)))
    if not files:
        logger.warning(f"No NIfTI files found in {nifti_path} with pattern {file_pattern}.")
        return
    logger.info(f"Converting {len(files)} files: is_label={is_label}, target_spacing={target_spacing_xyz}, workers={num_workers}, force={force}")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futs = {
            executor.submit(safe_convert_nifti_file, f, output_dir, is_label, target_spacing_xyz, True, force): f
            for f in files
        }
        for fut in tqdm(futs, desc=f"Converting {os.path.basename(nifti_path)}"):
            file = futs[fut]
            try:
                ok, msg = fut.result()
                if not ok:
                    logger.error(f"Failed to convert {file}: {msg}")
            except Exception as e:
                logger.error(f"Conversion process for {file} raised an unexpected error: {str(e)}")


def process_single_file(label_file, output_dir, sigma, decay_factor):
    try:
        filename = os.path.basename(label_file)
        label_data = np.load(label_file)

        if np.sum(label_data > 0) == 0:
            logger.warning(f"Label file {filename} has no foreground voxels, generating default weight map.")
            weight_map = np.ones_like(label_data, dtype=np.float32)
        else:
            binary_mask = (label_data > 0).astype(np.uint8)
            distance_map = distance_transform_edt(binary_mask == 0)
            weight_map = 1 + decay_factor * np.exp(-distance_map**2 / (2 * sigma**2))

        out = os.path.join(output_dir, filename)
        np.save(out, weight_map.astype(np.float32))
        return filename
    except Exception as e:
        logger.error(f"Error processing weight map for {label_file}: {str(e)}")
        return None


def generate_weight_maps(label_dir, output_dir, sigma=20, decay_factor=5, num_processes=4, num_workers=1):
    os.makedirs(output_dir, exist_ok=True)
    label_files = sorted([f for f in glob(os.path.join(label_dir, "*.npy")) if "_metadata" not in os.path.basename(f)])
    if not label_files:
        logger.warning(f"No label files found in {label_dir}")
        return

    actual_num_workers = num_workers if num_workers != -1 else (num_processes or (os.cpu_count() or 1))
    logger.info(f"Generating weight maps for {len(label_files)} label files using {actual_num_workers} workers/processes.")

    results = []
    if actual_num_workers == 1 or len(label_files) < actual_num_workers:
        for lf in tqdm(label_files):
            results.append(process_single_file(lf, output_dir, sigma, decay_factor))
    else:
        results = Parallel(n_jobs=actual_num_workers)(
            delayed(process_single_file)(lf, output_dir, sigma, decay_factor) for lf in tqdm(label_files)
        )
    ok = [r for r in results if r is not None]
    logger.info(f"Weight maps saved to {output_dir} ({len(ok)}/{len(label_files)} successful).")


def process_single_lung_mask(nifti_path, output_dir, target_spacing_xyz=None, force: bool = False,
                             gpu_semaphore: Optional[multiprocessing.synchronize.Semaphore] = None,
                             force_cpu: Optional[bool] = None,
                             lm_batch_env: Optional[int] = None):

    try:
        filename_base = os.path.basename(nifti_path).split('.')[0]
        out_path = os.path.join(output_dir, f"{filename_base}.npy")
        if os.path.exists(out_path) and not force:
            logger.debug(f"Lung mask {out_path} exists; skip (use force=True to overwrite).")
            return True, None

        nii_orig = nib.load(nifti_path)
        orig_affine = nii_orig.affine.copy()
        orig_shape = nii_orig.shape
        orig_spacing = np.sqrt(np.sum(orig_affine[:3, :3] ** 2, axis=0))
        orig_dir = (orig_affine[:3, :3] / np.maximum(orig_spacing, 1e-6)).tolist()
        orig_org = orig_affine[:3, 3].tolist()

        img_sitk = _load_and_preprocess_sitk(nifti_path, is_label=False)
        if img_sitk is None:
            return False, f"Failed to load {nifti_path}"
        img_lps = sitk.DICOMOrient(img_sitk, "LPS")  
        if target_spacing_xyz and len(target_spacing_xyz) == 3:
            target_spacing = tuple(float(s) for s in target_spacing_xyz)
            o_sz, o_sp = img_lps.GetSize(), img_lps.GetSpacing()
            new_size = [int(round(sz * sp / tsp)) for sz, sp, tsp in zip(o_sz, o_sp, target_spacing)]
            R = sitk.ResampleImageFilter()
            R.SetInterpolator(sitk.sitkLinear)
            R.SetDefaultPixelValue(-1024.0)
            R.SetOutputSpacing(target_spacing)
            R.SetSize(new_size)
            R.SetOutputDirection(img_lps.GetDirection())
            R.SetOutputOrigin(img_lps.GetOrigin())
            R.SetTransform(sitk.Transform())
            img_proc = R.Execute(img_lps)
        else:
            img_proc = img_lps

        seg = None
        try:
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            lm_batch = lm_batch_env if lm_batch_env is not None else int(os.getenv("LUNGMASK_BATCH", "1" if torch.cuda.is_available() else "8"))
            use_cpu = force_cpu if force_cpu is not None else ((not torch.cuda.is_available()) or os.getenv("LUNGMASK_DEVICE", "auto").lower() in ("cpu", "0", "false"))
            coarse_mm = float(os.getenv("LUNGMASK_COARSE_SPACING", "1.6"))
            coarse_mm = max(0.8, coarse_mm) 
            cur_sp = img_proc.GetSpacing()
            target_coarse_sp = (
                max(float(cur_sp[0]), coarse_mm),
                max(float(cur_sp[1]), coarse_mm),
                max(float(cur_sp[2]), coarse_mm),
            )
            if any(abs(target_coarse_sp[i] - float(cur_sp[i])) > 1e-6 for i in range(3)):
                o_sz = img_proc.GetSize()
                new_size = [int(round(sz * sp / tsp)) for sz, sp, tsp in zip(o_sz, cur_sp, target_coarse_sp)]
                Rpre = sitk.ResampleImageFilter()
                Rpre.SetInterpolator(sitk.sitkLinear)
                Rpre.SetDefaultPixelValue(-1024.0)
                Rpre.SetOutputSpacing(target_coarse_sp)
                Rpre.SetSize(new_size)
                Rpre.SetOutputDirection(img_proc.GetDirection())
                Rpre.SetOutputOrigin(img_proc.GetOrigin())
                Rpre.SetTransform(sitk.Transform())
                img_for_lm = Rpre.Execute(img_proc)
            else:
                img_for_lm = img_proc
            
            if not use_cpu and gpu_semaphore is not None:
                gpu_semaphore.acquire()
            try:
                lm = lungmask_model.apply(
                    img_for_lm,
                    batch_size=lm_batch,
                    volume_postprocessing=True,
                    force_cpu=use_cpu
                )
            finally:
                if not use_cpu and gpu_semaphore is not None:
                    gpu_semaphore.release()

            lm_bin = (lm > 0).astype(np.uint8)
            if img_for_lm is not img_proc:
                lm_img = sitk.GetImageFromArray(lm_bin)
                lm_img.CopyInformation(img_for_lm)
                Rback = sitk.ResampleImageFilter()
                Rback.SetInterpolator(sitk.sitkNearestNeighbor)
                Rback.SetDefaultPixelValue(0)
                Rback.SetOutputSpacing(img_proc.GetSpacing())
                Rback.SetSize(img_proc.GetSize())
                Rback.SetOutputDirection(img_proc.GetDirection())
                Rback.SetOutputOrigin(img_proc.GetOrigin())
                Rback.SetTransform(sitk.Transform())
                lm_img_back = Rback.Execute(lm_img)
                seg = sitk.GetArrayFromImage(lm_img_back).astype(np.uint8)
                del lm_img_back, lm_img
            else:
                seg = lm_bin
            del lm_bin
        except Exception as e:
            logger.warning(f"lungmask failed on {filename_base}: {e} – falling back to robust candidates")

        # Robust candidate on the SAME grid
        cand, body_mask = _air_body_candidates(img_proc)
        # Gate or fallback (guard seg is not None)
        # Robust decision: if lungmask exists but gating empties it (or is implausibly tiny),
        # fall back to the model‑free candidate computed on the same grid.
        if seg is not None and seg.sum() > 0:
            seg_gated = (seg > 0) & (body_mask > 0)
            # tiny threshold: <0.1% of volume
            tiny_thresh = 1e-3 * float(seg_gated.size)
            if seg_gated.sum() <= 0 or float(seg_gated.sum()) < tiny_thresh:
                # Prefer candidate if non‑empty; otherwise keep the original lungmask result
                if cand is not None and (cand > 0).sum() > 0:
                    seg = (cand > 0)
                else:
                    seg = seg_gated
            else:
                seg = seg_gated
        else:
            seg = (cand > 0)
        seg = _clear_border_xy_per_slice(seg)
        struct = generate_binary_structure(3, 1)
        lab, n = skimage_label(seg.astype(np.uint8), return_num=True)
        if n > 2:
            bc = np.bincount(lab.ravel())
            order = np.argsort(bc[1:])[::-1] + 1
            seg = np.isin(lab, order[:2])
        seg = binary_closing(seg, structure=struct, iterations=2)
        seg = binary_fill_holes(seg)
        seg = binary_dilation(seg, structure=struct, iterations=1) & (body_mask > 0)
        arr_hu = sitk.GetArrayFromImage(img_proc).astype(np.float32)
        struct = generate_binary_structure(3, 1)
        seg = (seg > 0) & (body_mask > 0)
        seg = _clear_border_xy_per_slice(seg)
        lab, n = skimage_label(seg.astype(np.uint8), return_num=True)
        if n > 0:
            comps = []
            for lbl in range(1, n + 1):
                comp = (lab == lbl)
                if not comp.any():
                    continue
                mean_hu = float(arr_hu[comp].mean())
                size = int(comp.sum())
                comps.append((size, mean_hu, lbl))

            # Prefer components with mean HU <= -300 (exclude mediastinum/soft tissue)
            air_like = [(s, l) for (s, m, l) in comps if m <= -300.0]
            if len(air_like) >= 2:
                air_like.sort(reverse=True)  # by size
                keep_labels = [air_like[0][1], air_like[1][1]]
            elif len(air_like) == 1:
                keep_labels = [air_like[0][1]]
                # add the largest remaining as second lung if present
                remaining = sorted([(s, l) for (s, m, l) in comps if l != keep_labels[0]], reverse=True)
                if remaining:
                    keep_labels.append(remaining[0][1])
            else:
                # no air-like comps → just keep two largest
                comps.sort(reverse=True)
                keep_labels = [t[2] for t in comps[:2]]

            seg = np.isin(lab, keep_labels)

        # Gentle morphology: close small holes, fill, then clamp to body
        seg = binary_closing(seg, structure=struct, iterations=1)
        seg = binary_fill_holes(seg)
        seg = seg & (body_mask > 0)
        seg = _clear_border_xy_per_slice(seg)
        lab, n = skimage_label(seg.astype(np.uint8), return_num=True)
        if n > 2:
            bc = np.bincount(lab.ravel())
            order = np.argsort(bc[1:])[::-1] + 1
            seg = np.isin(lab, order[:2])

        seg = seg.astype(np.uint8)

        np.save(out_path, seg)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        meta = {
            "original_filename": os.path.basename(nifti_path),
            "original_affine": orig_affine.tolist(),
            "original_spatial_shape": list(orig_shape),
            "original_spacing": orig_spacing.tolist(),
            "original_size": list(orig_shape),
            "original_direction": orig_dir,
            "original_origin": orig_org,
            "processed_spacing": list(img_proc.GetSpacing()),
            "processed_size": list(img_proc.GetSize()),
            "processed_direction": list(img_proc.GetDirection()),
            "processed_origin": list(img_proc.GetOrigin()),
            "numpy_array_shape_zyx": list(seg.shape),
            "is_label": True,
            "target_spacing_if_resampled": target_spacing_xyz if target_spacing_xyz else "N/A",
        }
        base = os.path.splitext(os.path.basename(out_path))[0]
        _save_meta(os.path.join(output_dir, base), meta)
        del seg, img_proc, img_lps, img_sitk
        gc.collect()
        return True, None

    except Exception as e:
        err = f"Error generating lung mask for {nifti_path}: {e}"
        logger.critical(err)
        return False, err


def generate_lung_masks(input_dir, output_dir, num_workers, target_spacing_xyz=None, force: bool = False,
                        lungmask_workers: Optional[int] = None,
                        gpu_concurrency: Optional[int] = None):
    """Generate lung masks from original NIfTI files with GPU-safe gating.

    - lungmask_workers: total parallel workers doing preprocessing + postprocessing
    - gpu_concurrency: max concurrent lungmask GPU inferences (1 recommended)
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(glob(os.path.join(input_dir, "*.nii.gz")))
    if not image_files:
        logger.warning(f"No NIfTI image files found in {input_dir}")
        return

    want_gpu = bool(torch.cuda.is_available())
    env_dev = os.getenv("LUNGMASK_DEVICE", "auto").lower()
    if env_dev in ("cpu", "0", "false"):
        want_gpu = False

    effective_workers = (
        int(lungmask_workers) if lungmask_workers is not None else (3 if want_gpu else max(1, int(num_workers)))
    )
    gpu_limit = int(gpu_concurrency) if gpu_concurrency is not None else (1 if want_gpu else 0)
    logger.info(f"Lungmask device={'cuda' if want_gpu else 'cpu'}; workers={effective_workers}; gpu_concurrency={gpu_limit if want_gpu else 'N/A'}.")
    gpu_sema = None
    if want_gpu and gpu_limit > 0:
        try:
            manager = multiprocessing.Manager()
            gpu_sema = manager.Semaphore(gpu_limit)
        except Exception:
            gpu_sema = None  # fallback: proceed without gating
    force_cpu = not want_gpu
    lm_batch_env = int(os.getenv("LUNGMASK_BATCH", "1" if want_gpu else "8"))

    if effective_workers == 1:
        results = []
        for nifti_path in tqdm(image_files, desc="Generating Lung Masks"):
            results.append(
                process_single_lung_mask(
                    nifti_path,
                    output_dir,
                    target_spacing_xyz,
                    force=force,
                    gpu_semaphore=gpu_sema,
                    force_cpu=force_cpu,
                    lm_batch_env=lm_batch_env,
                )
            )
    else:
        results = Parallel(
            n_jobs=effective_workers,
            backend="loky",
            prefer="processes",
            pre_dispatch="2*n_jobs",
        )(
            delayed(process_single_lung_mask)(
                nifti_path,
                output_dir,
                target_spacing_xyz,
                force,
                gpu_sema,
                force_cpu,
                lm_batch_env,
            )
            for nifti_path in tqdm(image_files, desc="Generating Lung Masks")
        )

    successful_count = sum(1 for r in results if isinstance(r, tuple) and r[0] or r is True)
    logger.info(f"Lung mask generation complete. {successful_count}/{len(image_files)} succeeded. Saved to {output_dir}")

def verify_single_file(file_info):
    try:
        image_path, label_path, lung_path = file_info
        results = {
            'filename': os.path.basename(image_path) if image_path else os.path.basename(label_path) if label_path else os.path.basename(lung_path),
            'has_image': False,
            'has_label': False,
            'has_lung': False,
            'image_shape': None,
            'label_shape': None,
            'lung_shape': None,
            'label_foreground': 0,
            'lung_foreground': 0,
            'errors': []
        }

        if image_path and os.path.exists(image_path):
            results['has_image'] = True
            image_data = np.load(image_path, allow_pickle=True, mmap_mode='r')
            results['image_shape'] = image_data.shape
            del image_data; gc.collect()

        if label_path and os.path.exists(label_path):
            results['has_label'] = True
            label_data = np.load(label_path, allow_pickle=True, mmap_mode='r')
            results['label_shape'] = label_data.shape
            results['label_foreground'] = np.sum(label_data > 0)
            del label_data; gc.collect()

        if lung_path and os.path.exists(lung_path):
            results['has_lung'] = True
            lung_data = np.load(lung_path, allow_pickle=True, mmap_mode='r')
            results['lung_shape'] = lung_data.shape
            results['lung_foreground'] = np.sum(lung_data > 0)
            del lung_data; gc.collect()

        return results
    except Exception as e:
        return {
            'filename': os.path.basename(image_path) if image_path else os.path.basename(label_path) if label_path else os.path.basename(lung_path),
            'errors': [str(e)]
        }


def verify_dataset_integrity(images_dir, labels_dir, lungs_dir=None, weights_dir=None, num_workers=-1):
    logger.info("Verifying dataset integrity...")
    image_files = {os.path.basename(f) for f in glob(os.path.join(images_dir, "*.npy")) if "_metadata" not in f}
    label_files = {os.path.basename(f) for f in glob(os.path.join(labels_dir, "*.npy")) if "_metadata" not in f}
    lung_files = {os.path.basename(f) for f in glob(os.path.join(lungs_dir, "*.npy")) if "_metadata" not in f} if lungs_dir else set()
    verification_tasks = []
    all_files = image_files.union(label_files).union(lung_files)
    for filename in all_files:
        image_path = os.path.join(images_dir, filename) if filename in image_files else None
        label_path = os.path.join(labels_dir, filename) if filename in label_files else None
        lung_path = os.path.join(lungs_dir, filename) if filename in lung_files else None
        if image_path or label_path or lung_path:
            verification_tasks.append((image_path, label_path, lung_path))

    logger.info(f"Starting parallel verification of {len(verification_tasks)} files using {num_workers} workers...")
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(verify_single_file)(task) for task in verification_tasks
    )

    issues = {'missing_pairs': [], 'shape_mismatches': [], 'empty_labels': [], 'tiny_labels': [], 'empty_lungs': [], 'errors': []}

    for r in results:
        filename = r['filename']
        if r.get('errors'):
            issues['errors'].append((filename, r['errors'])); continue

        if r['has_image'] and not r['has_label']:
            issues['missing_pairs'].append(f"{filename} (missing label)")
        elif r['has_label'] and not r['has_image']:
            issues['missing_pairs'].append(f"{filename} (missing image)")

        if r['has_image'] and r['has_label'] and r['image_shape'] != r['label_shape']:
            issues['shape_mismatches'].append((filename, r['image_shape'], r['label_shape']))

        if r['has_label']:
            if r['label_foreground'] == 0:
                issues['empty_labels'].append(filename)
            elif r['label_foreground'] < 1000:
                issues['tiny_labels'].append((filename, r['label_foreground']))

        if r['has_lung'] and r['lung_foreground'] == 0:
            issues['empty_lungs'].append(filename)

    logger.info("\nDataset Verification Results:")
    logger.info(f"Total files checked: {len(results)}")
    if issues['errors']:
        logger.error(f"\nFound {len(issues['errors'])} files with errors:")
        for filename, errs in issues['errors']:
            logger.error(f"  - {filename}: {errs}")
    if issues['missing_pairs']:
        logger.warning(f"\nFound {len(issues['missing_pairs'])} incomplete pairs:")
        for item in issues['missing_pairs']:
            logger.warning(f"  - {item}")
    if issues['shape_mismatches']:
        logger.warning(f"\nFound {len(issues['shape_mismatches'])} shape mismatches:")
        for filename, img_shape, label_shape in issues['shape_mismatches']:
            logger.warning(f"  - {filename}: Image {img_shape} vs Label {label_shape}")
    if issues['empty_labels']:
        logger.warning(f"\nFound {len(issues['empty_labels'])} empty label files:")
        for filename in issues['empty_labels']:
            logger.warning(f"  - {filename}")
    if issues['tiny_labels']:
        logger.warning(f"\nFound {len(issues['tiny_labels'])} labels with very few foreground voxels:")
        for filename, count in issues['tiny_labels']:
            logger.warning(f"  - {filename}: {count} voxels")
    if issues['empty_lungs']:
        logger.warning(f"\nFound {len(issues['empty_lungs'])} empty lung masks:")
        for filename in issues['empty_lungs']:
            logger.warning(f"  - {filename}")

    valid_count = len(results) - len(issues['missing_pairs']) - len(issues['shape_mismatches'])
    logger.info(f"\nValid paired samples: {valid_count}")
    return valid_count > 0

def calculate_average_spacing(nifti_files: List[str]) -> Optional[Tuple[float, float, float]]:
    if not nifti_files:
        return None
    total_spacing = np.array([0.0, 0.0, 0.0]); num_files = 0
    for file_path in tqdm(nifti_files, desc="Calculating Average Spacing"):
        try:
            nii_orig = nib.load(file_path)
            affine = nii_orig.affine
            spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
            total_spacing += spacing; num_files += 1
        except Exception as e:
            logger.warning(f"Could not read spacing from {file_path}: {e}")
    if num_files == 0:
        return None
    avg = total_spacing / num_files
    logger.info(f"Calculated average spacing: {avg.tolist()}")
    return tuple(avg)

def main():
    parser = argparse.ArgumentParser(description="Prepare ATM22 dataset for airway segmentation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to ATM22 dataset")
    parser.add_argument("--batch1", type=str, help="Name of TrainBatch1 directory")
    parser.add_argument("--batch2", type=str, help="Name of TrainBatch2 directory")
    parser.add_argument("--sigma", type=float, default=20, help="Sigma parameter for weight maps")
    parser.add_argument("--decay", type=float, default=5, help="Decay factor for weight maps")
    parser.add_argument("--generate_lungs", action="store_true", help="Generate lung masks")
    parser.add_argument("--force_lungs", action="store_true", help="Regenerate lung masks even if files exist")  # NEW
    parser.add_argument("--process_validation", action="store_true", help="Process validation data")
    parser.add_argument("--force_recreate", action="store_true", help="Force recreation of npy files")
    parser.add_argument("--verify_only", action="store_true", help="Only verify the existing dataset without conversion")
    parser.add_argument("--num_workers", type=int, default=-1, help="Workers for parallel tasks")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None, help="Target spacing (e.g., 1.2 1.2 1.2)")
    parser.add_argument("--lungmask_workers", type=int, default=None, help="Total workers for lungmask preprocessing/postprocessing")
    parser.add_argument("--lungmask_gpu_limit", type=int, default=None, help="Max concurrent GPU lungmask inferences (recommend 1)")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = os.path.join(dataset_path, "npy_files")
    images_output = os.path.join(output_path, "imagesTr")
    labels_output = os.path.join(output_path, "labelsTr")
    lungs_output  = os.path.join(output_path, "lungsTr")
    weights_output = os.path.join(output_path, "Wr1alpha0.2")
    val_output = os.path.join(dataset_path, "val")
    val_lungs_output = os.path.join(dataset_path, "val_lungs")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(val_lungs_output, exist_ok=True)
    os.makedirs(weights_output, exist_ok=True)

    if args.verify_only:
        logger.info("Verification-only mode")
        verify_dataset_integrity(images_output, labels_output, lungs_output, weights_output)
        return

    conversion_workers = args.num_workers if args.num_workers != -1 else (os.cpu_count() or 4)
    conversion_workers = min(conversion_workers, 8)

    # Auto target spacing if not provided
    target_spacing = args.target_spacing
    if target_spacing is None:
        logger.info("Target spacing not provided. Calculating average spacing from input NIfTI images...")
        all_image_files_to_process = []
        if args.batch1:
            p = os.path.join(dataset_path, args.batch1, "imagesTr")
            if os.path.exists(p): all_image_files_to_process.extend(glob(os.path.join(p, "*.nii.gz")))
        if args.batch2:
            p = os.path.join(dataset_path, args.batch2, "imagesTr")
            if os.path.exists(p): all_image_files_to_process.extend(glob(os.path.join(p, "*.nii.gz")))
        if args.process_validation:
            p = os.path.join(dataset_path, "imagesVal")
            if os.path.exists(p): all_image_files_to_process.extend(glob(os.path.join(p, "*.nii.gz")))
        if all_image_files_to_process:
            target_spacing = calculate_average_spacing(all_image_files_to_process)
            if target_spacing:
                logger.info(f"Using automatically calculated average spacing: {target_spacing}")
    else:
        logger.info(f"Using user-provided target spacing: {target_spacing}")

    # TrainBatch1
    if args.batch1:
        b1 = os.path.join(dataset_path, args.batch1)
        logger.info(f"Processing {b1}")
        images_path = os.path.join(b1, "imagesTr")
        if os.path.exists(images_path):
            convert_nifti_to_numpy(images_path, images_output, is_label=False, target_spacing_xyz=target_spacing,
                                   num_workers=conversion_workers, force=args.force_recreate)
        labels_path = os.path.join(b1, "labelsTr")
        if os.path.exists(labels_path):
            convert_nifti_to_numpy(labels_path, labels_output, is_label=True, target_spacing_xyz=target_spacing,
                                   num_workers=conversion_workers, force=args.force_recreate)

    # TrainBatch2
    if args.batch2:
        b2 = os.path.join(dataset_path, args.batch2)
        logger.info(f"Processing {b2}")
        images_path = os.path.join(b2, "imagesTr")
        if os.path.exists(images_path):
            convert_nifti_to_numpy(images_path, images_output, is_label=False, target_spacing_xyz=target_spacing,
                                   num_workers=conversion_workers, force=args.force_recreate)
        labels_path = os.path.join(b2, "labelsTr")
        if os.path.exists(labels_path):
            convert_nifti_to_numpy(labels_path, labels_output, is_label=True, target_spacing_xyz=target_spacing,
                                   num_workers=conversion_workers, force=args.force_recreate)

    # Validation
    if args.process_validation:
        val_images_path = os.path.join(dataset_path, "imagesVal")
        if os.path.exists(val_images_path):
            logger.info(f"Processing validation images from {val_images_path}")
            if args.generate_lungs:
                logger.info("Generating lung masks for validation data (from original NIfTI files)...")
                generate_lung_masks(
                    val_images_path,
                    val_lungs_output,
                    num_workers=args.num_workers,
                    target_spacing_xyz=target_spacing,
                    force=args.force_lungs,
                    lungmask_workers=args.lungmask_workers,
                    gpu_concurrency=args.lungmask_gpu_limit,
                )
            convert_nifti_to_numpy(val_images_path, val_output, is_label=False, target_spacing_xyz=target_spacing,
                                   num_workers=conversion_workers, force=args.force_recreate)
            logger.info("Validation data processing complete.")

    # Sanity checks on output dirs
    if not os.path.exists(images_output):
        logger.error(f"Images directory {images_output} does not exist. Exiting.")
        return
    if not os.path.exists(labels_output):
        logger.error(f"Labels directory {labels_output} does not exist. Exiting.")
        return

    # Training set lung masks (from original NIfTI dirs)
    if args.generate_lungs:
        logger.info("Generating lung masks for training set...")
        if args.batch1:
            generate_lung_masks(
                os.path.join(dataset_path, args.batch1, "imagesTr"),
                lungs_output,
                num_workers=args.num_workers,
                target_spacing_xyz=target_spacing,
                force=args.force_lungs,
                lungmask_workers=args.lungmask_workers,
                gpu_concurrency=args.lungmask_gpu_limit,
            )
        if args.batch2:
            generate_lung_masks(
                os.path.join(dataset_path, args.batch2, "imagesTr"),
                lungs_output,
                num_workers=args.num_workers,
                target_spacing_xyz=target_spacing,
                force=args.force_lungs,
                lungmask_workers=args.lungmask_workers,
                gpu_concurrency=args.lungmask_gpu_limit,
            )
        logger.info("Lung mask generation finished.")

    # Verify
    verify_dataset_integrity(images_output, labels_output, lungs_output, weights_output)

    if args.process_validation:
        logger.info(f"Prepared validation dataset is in: {val_output} and {val_lungs_output}")


if __name__ == "__main__":
    main()
