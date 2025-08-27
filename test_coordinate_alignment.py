#!/usr/bin/env python3
"""
Quick test script to verify coordinate system alignment in the prediction pipeline.
This will help us identify where the misalignment occurs.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from pathlib import Path
import json

def test_coordinate_consistency():
    """Test if we can round-trip coordinates correctly."""
    
    print("=== Coordinate System Alignment Test ===")
    
    # Test with a sample validation file
    val_npy_path = "./ATM22/val/ATM_281_0000.npy" 
    val_meta_path = "./ATM22/val/ATM_281_0000_meta.json"
    lung_npy_path = "./ATM22/val_lungs/ATM_281_0000.npy"
    original_nii_path = "./ATM22/imagesVal/ATM_281_0000.nii.gz"
    
    # Check if files exist
    for path_str, name in [(val_npy_path, "Validation NPY"), 
                           (val_meta_path, "Metadata JSON"),
                           (lung_npy_path, "Lung mask NPY"),
                           (original_nii_path, "Original NIfTI")]:
        if not Path(path_str).exists():
            print(f"❌ Missing: {name} at {path_str}")
            return False
        else:
            print(f"✅ Found: {name}")
    
    # Load the original NIfTI to get ground truth coordinates
    print("\n--- Loading Original NIfTI ---")
    original_nii = nib.load(original_nii_path)
    original_affine = original_nii.affine
    original_shape = original_nii.shape
    print(f"Original shape: {original_shape}")
    print(f"Original affine:\n{original_affine}")
    
    # Load processed data and metadata
    print("\n--- Loading Processed Data ---")
    processed_data = np.load(val_npy_path)
    lung_data = np.load(lung_npy_path)
    
    with open(val_meta_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Lung mask shape: {lung_data.shape}")
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # Check if the stored affine matches the original
    stored_original_affine = np.array(metadata['original_affine'])
    affine_diff = np.abs(original_affine - stored_original_affine).max()
    
    print(f"\n--- Affine Matrix Comparison ---")
    print(f"Max difference between stored and actual affine: {affine_diff:.6f}")
    if affine_diff < 1e-6:
        print("✅ Stored affine matrix matches original")
    else:
        print("❌ Affine matrix mismatch detected")
        print(f"Stored affine:\n{stored_original_affine}")
        print(f"Original affine:\n{original_affine}")
    
    # Test coordinate transformation
    print("\n--- Testing Coordinate Round-trip ---")
    
    # Create a simple test prediction (same shape as processed data)
    test_prediction = np.zeros_like(processed_data, dtype=np.uint8)
    # Add a small cube in the center as a test marker
    center = [s//2 for s in test_prediction.shape]
    test_prediction[center[0]-5:center[0]+5, 
                   center[1]-5:center[1]+5, 
                   center[2]-5:center[2]+5] = 1
    
    print(f"Test prediction shape: {test_prediction.shape}")
    print(f"Test prediction has {np.sum(test_prediction)} foreground voxels")
    
    # Simulate the resampling process that happens in SitkResampleToOriginalGridd
    print("\n--- Simulating Resampling Process ---")
    
    # Step 1: Create SimpleITK image from prediction (as in transforms.py)
    itk_prediction = sitk.GetImageFromArray(test_prediction)
    
    # Step 2: Set the processed coordinate system metadata
    processed_spacing = metadata['processed_spacing']
    processed_direction = metadata['processed_direction']
    processed_origin = metadata['processed_origin']
    
    itk_prediction.SetSpacing(processed_spacing)
    itk_prediction.SetDirection(processed_direction)
    itk_prediction.SetOrigin(processed_origin)
    
    # Step 3: Load reference image for target grid
    reference_sitk = sitk.ReadImage(original_nii_path)
    
    # Step 4: Resample to original grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_sitk.GetSpacing())
    resampler.SetOutputDirection(reference_sitk.GetDirection())
    resampler.SetOutputOrigin(reference_sitk.GetOrigin())
    resampler.SetSize(reference_sitk.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    resampled_prediction = resampler.Execute(itk_prediction)
    
    # Step 5: Convert back to numpy
    final_prediction = sitk.GetArrayFromImage(resampled_prediction)
    
    print(f"Resampled prediction shape: {final_prediction.shape}")
    print(f"Expected shape (original): {original_shape}")
    print(f"Resampled prediction has {np.sum(final_prediction)} foreground voxels")
    
    # Check if shapes match
    if final_prediction.shape == original_shape:
        print("✅ Shape consistency maintained")
    else:
        print("❌ Shape mismatch after resampling")
    
    # Test saving with correct affine (as in custom_save.py)
    print("\n--- Testing NIfTI Saving Process ---")
    
    # The data needs to be transposed from (Z,Y,X) to (X,Y,Z) for nibabel
    final_for_saving = np.transpose(final_prediction, (2, 1, 0))
    print(f"Data for NIfTI saving shape: {final_for_saving.shape}")
    
    # Create NIfTI image with original affine
    test_nifti = nib.Nifti1Image(final_for_saving, original_affine)
    
    # Save test file
    test_output_path = "./test_prediction_alignment.nii.gz"
    nib.save(test_nifti, test_output_path)
    
    print(f"✅ Test prediction saved to: {test_output_path}")
    print("\nTo verify alignment:")
    print(f"1. Load original: {original_nii_path}")  
    print(f"2. Load test prediction: {test_output_path}")
    print("3. Check if the central cube appears in the correct location")
    
    return True

if __name__ == "__main__":
    test_coordinate_consistency()