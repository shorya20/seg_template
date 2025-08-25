#!/usr/bin/env python3
"""
Test script to verify that predictions are properly aligned with original images
after fixing the Invertd transform for lung-based cropping.
"""

import numpy as np
import nibabel as nib
import os
from pathlib import Path
import sys

def check_alignment(pred_path, original_image_path, lung_mask_path=None):
    """
    Check if prediction is aligned with original image.
    
    Args:
        pred_path: Path to predicted segmentation
        original_image_path: Path to original image
        lung_mask_path: Optional path to lung mask
    
    Returns:
        dict: Alignment metrics
    """
    print(f"\n{'='*60}")
    print(f"Checking alignment for: {Path(pred_path).name}")
    print(f"{'='*60}")
    
    # Load files
    pred_nii = nib.load(pred_path)
    orig_nii = nib.load(original_image_path)
    
    pred_data = pred_nii.get_fdata()
    orig_data = orig_nii.get_fdata()
    
    # Check shapes
    print(f"\nShape comparison:")
    print(f"  Original image shape: {orig_data.shape}")
    print(f"  Prediction shape:     {pred_data.shape}")
    shape_match = orig_data.shape == pred_data.shape
    print(f"  Shapes match: {shape_match}")
    
    # Check affines
    print(f"\nAffine comparison:")
    affine_diff = np.abs(pred_nii.affine - orig_nii.affine).max()
    print(f"  Max affine difference: {affine_diff:.6f}")
    affine_match = affine_diff < 1e-5
    print(f"  Affines match: {affine_match}")
    
    # Check spacing
    orig_spacing = orig_nii.header.get_zooms()[:3]
    pred_spacing = pred_nii.header.get_zooms()[:3]
    print(f"\nSpacing comparison:")
    print(f"  Original spacing: {orig_spacing}")
    print(f"  Prediction spacing: {pred_spacing}")
    spacing_match = np.allclose(orig_spacing, pred_spacing, rtol=1e-5)
    print(f"  Spacings match: {spacing_match}")
    
    # Count non-zero voxels
    pred_voxels = np.sum(pred_data > 0)
    print(f"\nPrediction statistics:")
    print(f"  Non-zero voxels: {pred_voxels:,}")
    print(f"  Percentage of volume: {100 * pred_voxels / pred_data.size:.2f}%")
    
    # If lung mask provided, check overlap
    if lung_mask_path and os.path.exists(lung_mask_path):
        # Handle .npy files differently from NIfTI files
        if str(lung_mask_path).endswith('.npy'):
            lung_data = np.load(lung_mask_path)
        else:
        lung_nii = nib.load(lung_mask_path)
        lung_data = lung_nii.get_fdata()
        
        if lung_data.shape == pred_data.shape:
            # Calculate overlap between prediction and lung mask
            pred_in_lung = np.sum((pred_data > 0) & (lung_data > 0))
            pred_outside_lung = np.sum((pred_data > 0) & (lung_data == 0))
            
            print(f"\nLung mask overlap:")
            print(f"  Predictions inside lungs: {pred_in_lung:,} voxels")
            print(f"  Predictions outside lungs: {pred_outside_lung:,} voxels")
            if pred_voxels > 0:
                inside_ratio = pred_in_lung / pred_voxels
                print(f"  Percentage inside lungs: {100 * inside_ratio:.1f}%")
                
                if inside_ratio < 0.5:
                    print(f"  ⚠️ WARNING: Most predictions are outside lungs!")
                elif inside_ratio > 0.95:
                    print(f"  ✓ Good: Most predictions are inside lungs")
        else:
            print(f"\n⚠️ Lung mask shape {lung_data.shape} doesn't match prediction shape")
    
    # Overall assessment
    print(f"\n{'='*60}")
    print(f"ALIGNMENT ASSESSMENT:")
    if shape_match and affine_match and spacing_match:
        print(f"  ✓ PASS: Prediction is properly aligned with original image")
    else:
        print(f"  ✗ FAIL: Alignment issues detected")
        if not shape_match:
            print(f"    - Shape mismatch")
        if not affine_match:
            print(f"    - Affine mismatch")
        if not spacing_match:
            print(f"    - Spacing mismatch")
    
    if pred_voxels == 0:
        print(f"  ⚠️ WARNING: Prediction is empty (0 non-zero voxels)")
    
    print(f"{'='*60}\n")
    
    return {
        'shape_match': shape_match,
        'affine_match': affine_match,
        'spacing_match': spacing_match,
        'pred_voxels': pred_voxels,
        'aligned': shape_match and affine_match and spacing_match
    }


def main():
    """
    Main function to test alignment of predictions.
    """
    # Define paths
    base_dir = Path("/vol/bitbucket/ss2424/project/seg_template")
    
    # Example paths - adjust as needed
    pred_dir = base_dir / "outputs" / "DiceLoss_ATM22_8_5_16_AttentionUNet_prtrFalse_AdamW_128_b1_p3_0.0001_alpha0.3_r_d1.0r_l0.7ReduceLROnPlateau" / "preds"/ "ATM_281"
    orig_images_dir = base_dir / "ATM22" / "imagesVal"
    lung_masks_dir = base_dir / "ATM22" / "val_lungs"
    
    # Test a specific file (ATM_281_0000)
    test_case = "ATM_281"
    
    # Find prediction file
    pred_pattern = f"{test_case}_seg.nii.gz"
    pred_files = list(pred_dir.glob(f"**/{pred_pattern}"))
    
    if not pred_files:
        print(f"No prediction files found matching pattern: {pred_pattern}")
        print(f"Looking in: {pred_dir}")
        print("\nAvailable prediction files:")
        all_preds = list(pred_dir.glob("**/*.nii.gz"))
        for p in all_preds[:5]:  # Show first 5
            print(f"  - {p.relative_to(pred_dir)}")
        if len(all_preds) > 5:
            print(f"  ... and {len(all_preds) - 5} more")
        return
    
    # Use first matching prediction
    pred_path = pred_files[0]
    print(f"Found prediction: {pred_path}")
    
    # Find original image
    test_case = "ATM_281_0000"
    orig_path = orig_images_dir / f"{test_case}.nii.gz"
    if not orig_path.exists():
        print(f"Original image not found: {orig_path}")
        return
    
    # Find lung mask
    lung_path = lung_masks_dir / f"{test_case}.npy"
    if not lung_path.exists():
        print(f"Lung mask not found: {lung_path}")
        lung_path = None
    
    # Check alignment
    result = check_alignment(pred_path, orig_path, lung_path)
    
    # Summary
    if result['aligned']:
        print("\n✓ SUCCESS: Predictions are properly aligned with original images!")
        if result['pred_voxels'] > 0:
            print(f"  The fix has resolved the alignment issue.")
        else:
            print(f"  However, predictions are empty - may need to check model/inference.")
    else:
        print("\n✗ FAILURE: Alignment issues still present.")
        print("  The predictions are not properly aligned with original images.")
        print("  Further debugging needed.")


if __name__ == "__main__":
    main() 