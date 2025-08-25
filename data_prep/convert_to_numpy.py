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
from scipy.ndimage import binary_fill_holes
import traceback
from functools import partial
import multiprocessing
import torch


# Ensure we have SimpleITK and lungmask installed
try:
    import SimpleITK as sitk
    from lungmask import mask as lungmask_model
except ImportError:
    print("SimpleITK or lungmask is not installed. Please install it using 'pip install SimpleITK-SimpleElastix lungmask'.")
    sys.exit(1)
import numpy as np
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def _save_meta(base_path: str, meta: dict) -> None:
    """Write JSON next to output .npy. Converts tuples ➜ lists for JSON."""
    for k, v in meta.items():
        if isinstance(v, tuple):
            meta[k] = list(v)
    with open(base_path + "_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)


def _sigma_for_spacing(original: Tuple[float, float, float], target: Tuple[float, float, float]) -> List[float]:
    """Gaussian sigma (physical units) for anti‑aliasing when *down‑sampling*."""
    sigmas = [0.5 * (t / o) if t > o else 0.0 for t, o in zip(target, original)]
    return [s * o for s, o in zip(sigmas, original)]  # convert to physical units

def safe_convert_nifti_file(file, output_dir, is_label=False, target_spacing_xyz=None, antialias=True):
    """
    Safely convert and resample a NIfTI file to NumPy format, ensuring
    critical metadata ('original_affine', 'original_spatial_shape') is always saved.
    """
    try:
        # Load with nibabel FIRST to get the TRUE original affine and shape for final saving.
        # This is the most important step to fix the inconsistency.
        nii_orig = nib.load(file)
        original_affine = nii_orig.affine
        original_affine_nii = original_affine.tolist() # Convert to list for JSON
        original_shape_nii = nii_orig.shape
        
        # Extract PROPER direction and origin from the original affine matrix
        # The direction matrix should be normalized (unit vectors)
        original_spacing_from_affine = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
        original_direction = original_affine[:3, :3] / original_spacing_from_affine
        original_direction = original_direction.tolist()
        original_origin = original_affine[:3, 3].tolist()
    except Exception as e:
        logger.error(f"FATAL: Nibabel could not load {file} to get essential metadata: {e}")
        # If we can't get this, we cannot proceed.
        return False, f"Failed to load NIfTI header for {file}"

    try:
        file_path = file 
        fn = os.path.basename(file_path)
        stem = os.path.splitext(os.path.splitext(fn)[0])[0]
        out_npy = os.path.join(output_dir, f"{stem}.npy")
        if os.path.exists(out_npy):
            logger.debug("%s already exists – skipping", out_npy)
            return True, None

        # --- REFACTORED: Use centralized loading and preprocessing function ---
        itk_image_to_process = _load_and_preprocess_sitk(file_path, is_label=is_label)
        if itk_image_to_process is None:
            return False, f"Failed to load or preprocess {file_path}"
        # --- END REFACTORED SECTION ---

        # This section no longer queries the itk_image for metadata.
        # Instead, it exclusively uses the definitive metadata from the nibabel header.
        original_spacing_nii = original_spacing_from_affine.tolist()

        if target_spacing_xyz and len(target_spacing_xyz) == 3:
            target_spacing = tuple(float(s) for s in target_spacing_xyz)
            resample = sitk.ResampleImageFilter()
            if is_label:
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                resample.SetInterpolator(sitk.sitkLinear)
            default_pixel_value = 0.0 if is_label else -1024.0
            resample.SetDefaultPixelValue(default_pixel_value)
            new_size = [int(round(orig_sz * orig_sp / target_sp))
                        for orig_sz, orig_sp, target_sp in zip(original_shape_nii, original_spacing_nii, target_spacing)]
            resample.SetOutputSpacing(target_spacing)
            resample.SetSize(new_size)
            # Explicitly set the output orientation and origin from the original NIfTI file
            # to prevent SimpleITK from reorienting the image.
            resample.SetOutputDirection(np.array(original_direction).flatten().tolist())
            resample.SetOutputOrigin(original_origin)
            resample.SetTransform(sitk.Transform())
            resampled_itk_image = resample.Execute(itk_image_to_process)
        else:
            resampled_itk_image = itk_image_to_process

        final_data_zyx = sitk.GetArrayFromImage(resampled_itk_image)
        
        # Add a check to prevent saving empty arrays
        if final_data_zyx.size == 0:
            error_message = f"CRITICAL ERROR: Resampled image for {file_path} resulted in an empty array. Skipping save."
            logger.critical(error_message)
            return False, error_message
            
        final_dtype = np.float32 if not is_label else np.uint8
        final_data_zyx = final_data_zyx.astype(final_dtype)

        np.save(out_npy, final_data_zyx)
        metadata = {
            'original_filename': fn,
            'original_affine': original_affine_nii,
            'original_spatial_shape': original_shape_nii,
            'original_spacing': original_spacing_nii,
            'original_size': list(original_shape_nii),
            'original_direction': original_direction,
            'original_origin': original_origin,
            'processed_spacing': resampled_itk_image.GetSpacing(),
            'processed_size': resampled_itk_image.GetSize(),
            'processed_direction': resampled_itk_image.GetDirection(),
            'processed_origin': resampled_itk_image.GetOrigin(),
            'numpy_array_shape_zyx': final_data_zyx.shape,
            'is_label': is_label,
            'target_spacing_if_resampled': target_spacing_xyz if target_spacing_xyz else "N/A"
        }
        
        base_name = os.path.splitext(os.path.basename(out_npy))[0]
        _save_meta(os.path.join(output_dir, base_name), metadata)

        return True, None

    except Exception as e:
        error_message = f"CRITICAL ERROR during SimpleITK processing for {file_path}: {str(e)}\n{traceback.format_exc()}"
        logger.critical(error_message)
        return False, error_message

def convert_nifti_to_numpy(nifti_path, output_dir, file_pattern="*.nii.gz", is_label=False, target_spacing_xyz=None, num_workers=6):
    """
    Convert NIfTI files to NumPy arrays with validation and multiprocessing.
    
    Args:
        nifti_path (str): Path to directory containing NIfTI files.
        output_dir (str): Output directory for NumPy arrays.
        file_pattern (str): Pattern to match NIfTI files. Default is "*.nii.gz".
        is_label (bool): Whether the files are label masks.
        target_spacing (tuple): Target spacing for resampling.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(nifti_path, file_pattern))) # Ensure files is a list of strings
    if not files:
        logger.warning(f"No NIfTI files found in {nifti_path} with pattern {file_pattern}.")
        return
    
    logger.info(f"Found {len(files)} NIfTI files to convert in {nifti_path}.")
    logger.info(f"Converting {len(files)} NIfTI files to NumPy arrays: is_label={is_label}, target_spacing={target_spacing_xyz}, workers={num_workers}")

    # Use ProcessPoolExecutor for parallel conversion
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Ensure target_spacing_tuple is correctly passed
        future_to_file = {
            executor.submit(safe_convert_nifti_file, file, output_dir, is_label, target_spacing_xyz): file
            for file in files
        }
        for future in tqdm(future_to_file, desc=f"Converting {os.path.basename(nifti_path)}"):
            file = future_to_file[future]
            try:
                success, error_msg = future.result()
                if not success:
                    logger.error(f"Failed to convert {file}: {error_msg}")
            except Exception as e:
                logger.error(f"Conversion process for {file} raised an unexpected error: {str(e)}")
    

def process_single_file(label_file, output_dir, sigma, decay_factor):
    """Process a single label file to generate weight map."""
    try:
        filename = os.path.basename(label_file)
        label_data = np.load(label_file)
        
        # Check if label has any foreground voxels before calculating
        if np.sum(label_data > 0) == 0:
            logger.warning(f"Label file {filename} has no foreground voxels, generating default weight map.")
            weight_map = np.ones_like(label_data, dtype=np.float32)
        else:
            # Convert label data to binary mask (ensure it's binary)
            binary_mask = (label_data > 0).astype(np.uint8)
            
            # Calculate Euclidean distance transform - distance from background to foreground
            from scipy.ndimage import distance_transform_edt   
            distance_map = distance_transform_edt(binary_mask == 0)
            
            # Calculate weight map
            weight_map = 1 + decay_factor * np.exp(-distance_map**2 / (2 * sigma**2))
        
        # Save the weight map
        weight_map_filename = os.path.join(output_dir, filename)
        np.save(weight_map_filename, weight_map.astype(np.float32))
        return filename
    except Exception as e:
        logger.error(f"Error processing weight map for {label_file}: {str(e)}")
        return None

def generate_weight_maps(label_dir, output_dir, sigma=20, decay_factor=5, num_processes=4, num_workers=-1):
    """
    Generates weight maps from label files using multiprocessing.
    Args:
        label_dir (str): Directory containing label files (binary masks).
        output_dir (str): Directory to save the generated weight maps.
        sigma (float): Sigma for the Gaussian filter.
        decay_factor (float): Decay factor for the distance transform.
        num_processes (int): Number of processes to use for parallelization.
                             This will be overridden by num_workers if num_workers is not -1.
        num_workers (int): Number of workers to use for parallelization.
                           If -1, uses num_processes. If num_processes is also not set, it defaults to os.cpu_count().
    """
    os.makedirs(output_dir, exist_ok=True)
    label_files = sorted([f for f in glob(os.path.join(label_dir, "*.npy")) if "_metadata" not in os.path.basename(f)])

    if not label_files:
        logger.warning(f"No label files found in {label_dir}")
        return

    # Determine the number of workers to use
    if num_workers != -1:
        actual_num_workers = num_workers
    elif num_processes is not None: # num_processes might be the explicitly passed value or default
        actual_num_workers = num_processes
    else:
        actual_num_workers = os.cpu_count() or 1 # Fallback to 1 if os.cpu_count() is None

    logger.info(f"Generating weight maps for {len(label_files)} label files using {actual_num_workers} workers/processes.")

    # Using joblib's Parallel for consistency, though multiprocessing.Pool was also fine.
    # We will stick to the existing structure of process_single_file and adapt it slightly if needed.
    # The `process_single_file` function needs to be standalone or picklable.

    # Create a partial function with fixed arguments for sigma and decay_factor
    # process_func = partial(process_single_file, output_dir=output_dir, sigma=sigma, decay_factor=decay_factor)

    results = []
    # Sequential processing for easier debugging if actual_num_workers is 1 or for small N
    if actual_num_workers == 1 or len(label_files) < actual_num_workers : # Simple heuristic
        logger.info("Processing weight maps sequentially.")
        for label_file in tqdm(label_files):
            results.append(process_single_file(label_file, output_dir, sigma, decay_factor))
    else:
        logger.info(f"Processing weight maps in parallel with {actual_num_workers} workers.")
        # Parallel processing using joblib
        # Ensure process_single_file is compatible (it should be as it takes simple args)
        results = Parallel(n_jobs=actual_num_workers)(
            delayed(process_single_file)(label_file, output_dir, sigma, decay_factor)
            for label_file in tqdm(label_files)
        )
    
    successful = [r for r in results if r is not None]
    logger.info(f"Weight maps generated and saved to {output_dir} ({len(successful)}/{len(label_files)} successful)")

def process_single_lung_mask(nifti_path, output_dir, target_spacing_xyz=None):
    """
    Generates a robust lung mask from an original NIfTI CT image file using the lungmask library.
    Now uses the same processing pipeline as images/labels to ensure proper coordinate alignment,
    and includes robust percentile normalization for outlier cases.
    """
    try:
        filename_base = os.path.basename(nifti_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{filename_base}.npy")

        if os.path.exists(output_path):
            logger.debug(f"Lung mask {output_path} already exists, skipping generation.")
            return True

        # STEP 1: Load with nibabel to get original coordinate system info
        try:
            nii_orig = nib.load(nifti_path)
            original_affine = nii_orig.affine.copy()
            original_affine_nii = original_affine.tolist()
            original_shape_nii = nii_orig.shape
            original_spacing_from_affine = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
            original_direction = original_affine[:3, :3] / original_spacing_from_affine
            original_origin = original_affine[:3, 3]
        except Exception as e:
            logger.error(f"FATAL: Nibabel could not load {nifti_path} to get essential metadata: {e}")
            return False, f"Failed to load NIfTI header for {nifti_path}"

        # --- REFACTORED: Use centralized loading and preprocessing ---
        image_to_process = _load_and_preprocess_sitk(nifti_path, is_label=False)
        if image_to_process is None:
            return False, f"Failed to load or preprocess {nifti_path} for lungmask"
        # --- END REFACTORED SECTION ---

        # STEP 4: Apply resampling (same as safe_convert_nifti_file)
        if target_spacing_xyz and len(target_spacing_xyz) == 3:
            target_spacing = tuple(float(s) for s in target_spacing_xyz)
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetDefaultPixelValue(-1024.0)
            # Use nibabel-derived metadata for resampling calculation
            orig_size = original_shape_nii
            orig_spacing = original_spacing_from_affine
            new_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)]
            resample.SetOutputSpacing(target_spacing)
            resample.SetSize(new_size)
            # Explicitly set the output orientation and origin from the original NIfTI file.
            resample.SetOutputDirection(np.array(original_direction).flatten().tolist())
            resample.SetOutputOrigin(original_origin.tolist())
            resample.SetTransform(sitk.Transform())
            resampled_image = resample.Execute(image_to_process)
        else:
            resampled_image = image_to_process

        # --- STEP 5: Robust Normalization for Outlier Cases ---
        stats = sitk.StatisticsImageFilter()
        stats.Execute(resampled_image)
        img_min, img_max = stats.GetMinimum(), stats.GetMaximum()
        
        image_for_masking = resampled_image
        # Heuristic to detect outliers: min value is not typical for air, or max is very high
        if img_min > -500 or img_max > 3000:
            fn = os.path.basename(nifti_path)
            logger.warning(f"File {fn} has an unusual intensity range ({img_min:.2f}, {img_max:.2f}) after standard processing. Applying robust percentile normalization for lungmask.")
            
            np_image = sitk.GetArrayFromImage(resampled_image)
            non_bg_voxels = np_image[np_image > np_image.min()]
            if non_bg_voxels.size > 0:
                p1 = np.percentile(non_bg_voxels, 1)
                p99 = np.percentile(non_bg_voxels, 99)
                
                np_image = np.clip(np_image, p1, p99)
                
                min_val, max_val = np_image.min(), np_image.max()
                if max_val > min_val:
                    np_image = (np_image - min_val) / (max_val - min_val)  # Scale to [0, 1]
                    np_image = (np_image * 1600) - 1000  # Scale to [-1000, 600]
                
                normalized_sitk_image = sitk.GetImageFromArray(np_image.astype(np.float32))
                normalized_sitk_image.CopyInformation(resampled_image)
                image_for_masking = normalized_sitk_image
        
        # --- STEP 6: Run lungmask on the correctly processed image ---
        segmentation_np = lungmask_model.apply(image_for_masking)

        if np.sum(segmentation_np) == 0:
            logger.warning(f"Lung mask for {nifti_path} was generated empty by the model. Attempting to create a fallback body mask.")
            try:
                # Fallback: create a body mask by thresholding at -500 HU
                body_mask_np = sitk.GetArrayFromImage(image_for_masking) > -500

                # NEW: Add morphological opening to sever small connections between body and scanner bed
                from scipy.ndimage import binary_opening
                # Use a 3x3x3 structuring element. You might need to adjust the size and iterations.
                body_mask_np = binary_opening(body_mask_np, structure=np.ones((3,3,3)), iterations=2)

                # Find the largest connected component, which should be the patient's body
                labeled_mask, num_features = skimage_label(body_mask_np, return_num=True)
                if num_features > 0:
                    component_sizes = np.bincount(labeled_mask.ravel())
                    if len(component_sizes) > 1:
                        # Ignore background label 0
                        largest_component_label = component_sizes[1:].argmax() + 1
                        # Create a mask of only the largest component
                        body_mask_np = (labeled_mask == largest_component_label)
                    else:
                        body_mask_np = np.zeros_like(body_mask_np, dtype=bool)
                else:
                    body_mask_np = np.zeros_like(body_mask_np, dtype=bool)

                # Fill holes within the largest component to create a solid body mask
                final_body_mask = binary_fill_holes(body_mask_np).astype(np.uint8)

                # FINAL SANITY CHECK: If the resulting mask is still huge, it's invalid.
                # A body mask should not occupy more than ~50% of the total volume.
                total_voxels = np.prod(final_body_mask.shape)
                mask_voxels = np.sum(final_body_mask)
                if (mask_voxels / total_voxels) > 0.5:
                    logger.error(f"Fallback for {nifti_path} created an invalid, oversized mask ({mask_voxels} voxels). Discarding and saving an empty mask.")
                    final_body_mask = np.zeros_like(final_body_mask, dtype=np.uint8)

                # Check if fallback was successful
                if np.sum(final_body_mask) > 0:
                    logger.info(f"Successfully created a fallback body mask for {nifti_path} with {np.sum(final_body_mask)} voxels.")
                    segmentation_np = final_body_mask
                else:
                    logger.error(f"Fallback body mask generation resulted in an empty mask for {nifti_path}. The file will be saved with an empty mask.")
            except Exception as fallback_e:
                logger.error(f"Error during fallback mask generation for {nifti_path}: {fallback_e}. The file will be saved with an empty mask.")

        # --- STEP 7: Align mask with image and save ---
        # Create a SimpleITK image from the numpy mask array.
        mask_sitk = sitk.GetImageFromArray(segmentation_np.astype(np.uint8))
        
        # CRITICAL: Copy the spatial information from the processed CT image to the mask.
        # This ensures the mask and the image are in the exact same spatial grid.
        mask_sitk.CopyInformation(resampled_image)
        
        # Now, get the numpy array from the correctly aligned SimpleITK image.
        final_mask = sitk.GetArrayFromImage(mask_sitk)
        
        # Save the correctly aligned mask array.
        np.save(output_path, final_mask)
        
        # The metadata now accurately reflects the data being saved.
        metadata = {
            'original_filename': os.path.basename(nifti_path),
            'original_affine': original_affine_nii,
            'original_spatial_shape': original_shape_nii,
            'original_spacing': original_spacing_from_affine.tolist(),
            'original_size': list(original_shape_nii),
            'original_direction': original_direction.tolist(),
            'original_origin': original_origin.tolist(),
            'processed_spacing': list(resampled_image.GetSpacing()),
            'processed_size': list(resampled_image.GetSize()),
            'processed_direction': list(resampled_image.GetDirection()),
            'processed_origin': list(resampled_image.GetOrigin()),
            'numpy_array_shape_zyx': final_mask.shape,
            'is_label': True,
            'target_spacing_if_resampled': target_spacing_xyz if target_spacing_xyz else "N/A"
        }
        
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        _save_meta(os.path.join(output_dir, base_name), metadata)

        return True, None

    except Exception as e:
        error_message = f"CRITICAL ERROR during lung mask generation for {nifti_path}: {str(e)}\n{traceback.format_exc()}"
        logger.critical(error_message)
        return False, error_message

def generate_lung_masks(input_dir, output_dir, num_workers, target_spacing_xyz=None):
    """
    Generates lung masks from original NIfTI CT images using the lungmask library.
    The input_dir should point to the directory of original .nii.gz files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # The input directory should contain the original NIfTI files
    image_files = sorted(glob(os.path.join(input_dir, "*.nii.gz")))
    if not image_files:
        logger.warning(f"No NIfTI image files found in {input_dir}")
        return

    # NEW: Check for GPU and adjust worker count to prevent CUDA OOM with lungmask
    effective_workers = num_workers
    if torch.cuda.is_available():
        logger.info(f"GPU detected. Forcing serial processing (num_workers=2) for lung mask generation to prevent CUDA OOM issues from multiple parallel processes.")
        effective_workers = 2
    else:
        logger.info(f"No GPU detected. Using specified {num_workers} workers for CPU-based processing.")

    logger.info(f"Generating lung masks for {len(image_files)} image files using {effective_workers} worker(s)")
    
    # Process each file in parallel
    results = Parallel(n_jobs=effective_workers)(
        delayed(process_single_lung_mask)(nifti_path, output_dir, target_spacing_xyz) 
        for nifti_path in tqdm(image_files, desc="Generating Lung Masks")
    )
    
    # Fix: Handle tuple returns from process_single_lung_mask
    successful_count = sum(1 for r in results if isinstance(r, tuple) and r[0] or r is True)
    
    logger.info(f"Lung mask generation complete. {successful_count}/{len(image_files)} files processed successfully. Files saved to {output_dir}")

def verify_single_file(file_info):
    """Verify a single file and return its verification results."""
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

        # Check image
        if image_path and os.path.exists(image_path):
            results['has_image'] = True
            image_data = np.load(image_path, allow_pickle=True, mmap_mode='r')
            results['image_shape'] = image_data.shape
            del image_data  # Free memory
            gc.collect()

        # Check label
        if label_path and os.path.exists(label_path):
            results['has_label'] = True
            label_data = np.load(label_path, allow_pickle=True, mmap_mode='r')
            results['label_shape'] = label_data.shape
            results['label_foreground'] = np.sum(label_data > 0)
            del label_data
            gc.collect()

        # Check lung mask if provided
        if lung_path and os.path.exists(lung_path):
            results['has_lung'] = True
            lung_data = np.load(lung_path, allow_pickle=True, mmap_mode='r')
            results['lung_shape'] = lung_data.shape
            results['lung_foreground'] = np.sum(lung_data > 0)
            del lung_data
            gc.collect()

        return results
    except Exception as e:
        return {
            'filename': os.path.basename(image_path) if image_path else os.path.basename(label_path) if label_path else os.path.basename(lung_path),
            'errors': [str(e)]
        }

def verify_dataset_integrity(images_dir, labels_dir, lungs_dir=None, weights_dir=None, num_workers=-1):
    """Verify dataset integrity with parallel processing."""
    logger.info("Verifying dataset integrity...")
    
    # Get lists of files
    image_files = {os.path.basename(f) for f in glob(os.path.join(images_dir, "*.npy")) if "_metadata" not in f}
    label_files = {os.path.basename(f) for f in glob(os.path.join(labels_dir, "*.npy")) if "_metadata" not in f}
    lung_files = {os.path.basename(f) for f in glob(os.path.join(lungs_dir, "*.npy")) if "_metadata" not in f} if lungs_dir else set()
    
    # Create verification tasks
    verification_tasks = []
    all_files = image_files.union(label_files).union(lung_files)
    
    for filename in all_files:
        image_path = os.path.join(images_dir, filename) if filename in image_files else None
        label_path = os.path.join(labels_dir, filename) if filename in label_files else None
        lung_path = os.path.join(lungs_dir, filename) if filename in lung_files else None
        if image_path or label_path or lung_path:  # Only add if at least one file exists
            verification_tasks.append((image_path, label_path, lung_path))
    
    # Process verification tasks in parallel
    logger.info(f"Starting parallel verification of {len(verification_tasks)} files using {num_workers} workers...")
    results = Parallel(n_jobs=num_workers, verbose=1)(
        delayed(verify_single_file)(task) for task in verification_tasks
    )
    
    # Analyze results
    issues = {
        'missing_pairs': [],
        'shape_mismatches': [],
        'empty_labels': [],
        'tiny_labels': [],
        'empty_lungs': [],
        'errors': []
    }
    
    for result in results:
        filename = result['filename']
        
        # Check for errors
        if result.get('errors'):
            issues['errors'].append((filename, result['errors']))
            continue
            
        # Check for missing pairs
        if result['has_image'] and not result['has_label']:
            issues['missing_pairs'].append(f"{filename} (missing label)")
        elif result['has_label'] and not result['has_image']:
            issues['missing_pairs'].append(f"{filename} (missing image)")
            
        # Check shapes
        if result['has_image'] and result['has_label']:
            if result['image_shape'] != result['label_shape']:
                issues['shape_mismatches'].append(
                    (filename, result['image_shape'], result['label_shape'])
                )
                
        # Check label foreground
        if result['has_label']:
            if result['label_foreground'] == 0:
                issues['empty_labels'].append(filename)
            elif result['label_foreground'] < 1000:
                issues['tiny_labels'].append((filename, result['label_foreground']))
                
        # Check lung mask foreground
        if result['has_lung'] and result['lung_foreground'] == 0:
            issues['empty_lungs'].append(filename)
    
    # Log results
    logger.info("\nDataset Verification Results:")
    logger.info(f"Total files checked: {len(results)}")
    
    if issues['errors']:
        logger.error(f"\nFound {len(issues['errors'])} files with errors:")
        for filename, errors in issues['errors']:
            logger.error(f"  - {filename}: {errors}")
            
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
    
    return valid_count > 0  # Return True if there are valid samples

def _load_and_preprocess_sitk(nifti_path: str, is_label: bool = False) -> sitk.Image | None:
    """
    Loads and preprocesses a NIfTI file into a SimpleITK image, applying HU correction if it's not a label.
    This centralized function has been REWRITTEN to use nibabel for loading to preserve the original
    coordinate system and prevent SimpleITK's default reorientation. This is critical for data alignment.
    """
    try:
        # STEP 1: Load with nibabel to get the raw data and correct original affine
        nii_orig = nib.load(nifti_path)
        data_np = np.asanyarray(nii_orig.dataobj) # Efficiently get data array
        original_affine = nii_orig.affine

        # STEP 2: Decompose affine to get spacing, direction, and origin
        spacing = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
        direction_matrix = original_affine[:3, :3] / spacing
        origin = original_affine[:3, 3]

        # STEP 3: Create a SimpleITK image from the numpy data and set its metadata explicitly
        img_sitk = sitk.GetImageFromArray(data_np.transpose()) # Use transpose to match sitk's ZYX axis order
        img_sitk.SetSpacing(spacing.tolist())
        img_sitk.SetDirection(direction_matrix.flatten().tolist())
        img_sitk.SetOrigin(origin.tolist())

        # STEP 4: Apply intensity correction logic (HU values) if it's not a label
        if is_label:
            return img_sitk

        # Cast to float32 for calculations
        itk_image_to_process = sitk.Cast(img_sitk, sitk.sitkFloat32)

        # Check for rescale slope and intercept in NIfTI header metadata
        if nii_orig.header.get_slope_inter():
            slope, intercept = nii_orig.header.get_slope_inter()
            if slope is not None and intercept is not None and (slope != 1.0 or intercept != 0.0):
                logger.info(f"Applying rescale slope ({slope}) and intercept ({intercept}) from NIfTI header for {os.path.basename(nifti_path)}.")
                itk_image_to_process = itk_image_to_process * slope + intercept
                return itk_image_to_process

        # Fallback for integer-type images missing explicit rescale tags
        if img_sitk.GetPixelIDValue() in (sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkUInt8, sitk.sitkInt8):
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(img_sitk)
            # If the minimum value is non-negative, it's likely raw data that needs a standard offset
            if stats_filter.GetMinimum() >= 0:
                fn = os.path.basename(nifti_path)
                logger.info(f"File {fn} appears to be raw scanner data (no rescale tags). Applying common HU intercept of -1024.")
                itk_image_to_process -= 1024.0
        
        return itk_image_to_process

    except Exception as e:
        logger.error(f"CRITICAL ERROR during nibabel/SITK loading/preprocessing for {nifti_path}: {str(e)}\n{traceback.format_exc()}")
        return None
def calculate_average_spacing(nifti_files: List[str]) -> Optional[Tuple[float, float, float]]:
    """
    Calculates the average voxel spacing from a list of NIfTI files.

    Args:
        nifti_files (List[str]): A list of paths to NIfTI files.

    Returns:
        Optional[Tuple[float, float, float]]: A tuple containing the average
        spacing along each axis (x, y, z), or None if no files are provided.
    """
    if not nifti_files:
        return None

    total_spacing = np.array([0.0, 0.0, 0.0])
    num_files = 0

    for file_path in tqdm(nifti_files, desc="Calculating Average Spacing"):
        try:
            nii_orig = nib.load(file_path)
            affine = nii_orig.affine
            spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
            total_spacing += spacing
            num_files += 1
        except Exception as e:
            logger.warning(f"Could not read spacing from {file_path}: {e}")

    if num_files == 0:
        return None

    average_spacing = total_spacing / num_files
    logger.info(f"Calculated average spacing: {average_spacing.tolist()}")
    return tuple(average_spacing)

def main():
    parser = argparse.ArgumentParser(description="Prepare ATM22 dataset for airway segmentation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to ATM22 dataset")
    parser.add_argument("--batch1", type=str, help="Name of TrainBatch1 directory")
    parser.add_argument("--batch2", type=str, help="Name of TrainBatch2 directory")
    parser.add_argument("--sigma", type=float, default=20, help="Sigma parameter for weight maps")
    parser.add_argument("--decay", type=float, default=5, help="Decay factor for weight maps")
    parser.add_argument("--generate_lungs", action="store_true", help="Generate lung masks")
    parser.add_argument("--process_validation", action="store_true", help="Process validation data")
    parser.add_argument("--force_recreate", action="store_true", help="Force recreation of npy files")
    parser.add_argument("--verify_only", action="store_true", help="Only verify the existing dataset without conversion")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of worker processes to use for parallel tasks. Defaults to all available CPUs for joblib, or specific defaults for other parallelized functions.")
    parser.add_argument("--threshold", type=float, default=-1000, help="Threshold for lung mask generation (in raw intensity units)")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None, 
                        help="Target spacing for resampling (e.g., 1.2 1.2 1.2). No resampling if None.")
    args = parser.parse_args()
    
    # Check if dataset path exists
    dataset_path = args.dataset_path
    parent_directory = os.path.dirname(dataset_path)
    output_path = os.path.join(dataset_path, "npy_files")
    
    # Create output directories
    images_output = os.path.join(output_path, "imagesTr")
    labels_output = os.path.join(output_path, "labelsTr")
    lungs_output = os.path.join(output_path, "lungsTr")
    weights_output = os.path.join(output_path, "Wr1alpha0.2")
    
    val_output = os.path.join(dataset_path, "val")
    val_lungs_output = os.path.join(dataset_path, "val_lungs")
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(val_lungs_output, exist_ok=True)
    os.makedirs(weights_output, exist_ok=True)
    
    # Verify mode - only check existing data
    if args.verify_only:
        logger.info("Running in verification-only mode")
        verify_dataset_integrity(images_output, labels_output, lungs_output, weights_output)
        return
    
    conversion_workers = args.num_workers if args.num_workers != -1 else os.cpu_count() or 4
    conversion_workers = min(conversion_workers, 8) # Cap at 8 to avoid excessive overhead
    
    # --- Automatic Target Spacing Calculation ---
    target_spacing = args.target_spacing
    if target_spacing is None:
        logger.info("Target spacing not provided. Calculating average spacing from all input NIfTI images...")
        all_image_files_to_process = []
        
        # Collect potential training image files from batch 1
        if args.batch1:
            images_path_b1 = os.path.join(dataset_path, args.batch1, "imagesTr")
            if os.path.exists(images_path_b1):
                all_image_files_to_process.extend(glob(os.path.join(images_path_b1, "*.nii.gz")))
        
        # Collect potential training image files from batch 2
        if args.batch2:
            images_path_b2 = os.path.join(dataset_path, args.batch2, "imagesTr")
            if os.path.exists(images_path_b2):
                all_image_files_to_process.extend(glob(os.path.join(images_path_b2, "*.nii.gz")))
        
        # Collect potential validation image files
        if args.process_validation:
            val_images_path = os.path.join(dataset_path, "imagesVal")
            if os.path.exists(val_images_path):
                all_image_files_to_process.extend(glob(os.path.join(val_images_path, "*.nii.gz")))
                
        if all_image_files_to_process:
            target_spacing = calculate_average_spacing(all_image_files_to_process)
            if target_spacing:
                 logger.info(f"Using automatically calculated average spacing for resampling: {target_spacing}")
        else:
            logger.warning("No NIfTI files found to calculate average spacing. Resampling will be skipped.")
    else:
        logger.info(f"Using user-provided target spacing: {target_spacing}")

    # Process TrainBatch1
    if args.batch1:
        batch1_path = os.path.join(dataset_path, args.batch1)
        logger.info(f"Processing {batch1_path}")
        
        # Convert images
        images_path = os.path.join(batch1_path, "imagesTr")
        if os.path.exists(images_path):
            convert_nifti_to_numpy(images_path, images_output, is_label=False, target_spacing_xyz=target_spacing, num_workers=conversion_workers)
        
        # Convert labels
        labels_path = os.path.join(batch1_path, "labelsTr")
        if os.path.exists(labels_path):
            convert_nifti_to_numpy(labels_path, labels_output, is_label=True, target_spacing_xyz=target_spacing, num_workers=conversion_workers)

    # Process TrainBatch2
    if args.batch2:
        batch2_path = os.path.join(dataset_path, args.batch2)
        logger.info(f"Processing {batch2_path}")
        
        # Convert images
        images_path = os.path.join(batch2_path, "imagesTr")
        if os.path.exists(images_path):
            convert_nifti_to_numpy(images_path, images_output, is_label=False, target_spacing_xyz=target_spacing, num_workers=conversion_workers)

        # Convert labels
        labels_path = os.path.join(batch2_path, "labelsTr")
        if os.path.exists(labels_path):
            convert_nifti_to_numpy(labels_path, labels_output, is_label=True, target_spacing_xyz=target_spacing, num_workers=conversion_workers)

    # Process validation data
    if args.process_validation:
        val_images_path = os.path.join(dataset_path, "imagesVal")
        if os.path.exists(val_images_path):
            logger.info(f"Processing validation images from {val_images_path}")
            
            # Generate lung masks for validation data from original NIfTI files FIRST
            if args.generate_lungs:
                logger.info("Generating lung masks for validation data from original NIfTI files...")
                generate_lung_masks(val_images_path, val_lungs_output, num_workers=args.num_workers, target_spacing_xyz=target_spacing)

            # Then, apply the conversion/resampling pipeline to the NIfTI images
            convert_nifti_to_numpy(val_images_path, val_output, is_label=False, target_spacing_xyz=target_spacing, num_workers=conversion_workers)
            
            logger.info(f"Validation data processing complete.")
    
    # Check if directories exist before proceeding
    if not os.path.exists(images_output):
        logger.error(f"Images directory {images_output} does not exist. Exiting.")
        return
    
    if not os.path.exists(labels_output):
        logger.error(f"Labels directory {labels_output} does not exist. Exiting.")
        return
    
    # Generate lung masks if requested, now from the original NIfTI directories
    if args.generate_lungs:
        logger.info("Generating lung masks for training set from original NIfTI files...")
        # Assume images are in batch1_path/imagesTr and batch2_path/imagesTr
        if args.batch1:
            generate_lung_masks(os.path.join(dataset_path, args.batch1, "imagesTr"), lungs_output, num_workers=args.num_workers, target_spacing_xyz=target_spacing)
        if args.batch2:
            generate_lung_masks(os.path.join(dataset_path, args.batch2, "imagesTr"), lungs_output, num_workers=args.num_workers, target_spacing_xyz=target_spacing)
        
        logger.info("Lung mask generation finished.")

    # Generate weight maps if labels are available
    # if os.path.exists(labels_output) and any(glob(os.path.join(labels_output, "*.npy"))):
    #     logger.info("Generating weight maps...")
    #     generate_weight_maps(labels_output, weights_output, sigma=args.sigma, decay_factor=args.decay, num_workers=num_workers)
    #     logger.info("Weight map generation finished.")
    # else:
    #     logger.info("Label directory for training not found or empty, skipping weight map generation.")
    
    # Run verification to check dataset integrity
    verify_dataset_integrity(images_output, labels_output, lungs_output, weights_output)
    
    if args.process_validation:
        logger.info(f"Prepared validation dataset is in: {val_output} and {val_lungs_output}")

if __name__ == "__main__":
    main()