#!/usr/bin/env python3
"""
Script to verify that predictions align correctly with original NIfTI images.
"""
import nibabel as nib
import numpy as np
import json
from pathlib import Path
import argparse


def verify_alignment(original_nii_path, prediction_nii_path, metadata_json_path):
    """
    Verify that a prediction aligns with the original NIfTI image.
    
    Args:
        original_nii_path: Path to original NIfTI file
        prediction_nii_path: Path to predicted segmentation NIfTI file
        metadata_json_path: Path to metadata JSON file
    """
    print(f"\nVerifying alignment for:")
    print(f"  Original: {original_nii_path}")
    print(f"  Prediction: {prediction_nii_path}")
    print(f"  Metadata: {metadata_json_path}")
    
    # Load original NIfTI
    original_nii = nib.load(original_nii_path)
    original_data = original_nii.get_fdata()
    original_affine = original_nii.affine
    
    # Load prediction NIfTI
    pred_nii = nib.load(prediction_nii_path)
    pred_data = pred_nii.get_fdata()
    pred_affine = pred_nii.affine
    
    # Load metadata
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Check shapes
    print(f"\nShape comparison:")
    print(f"  Original shape: {original_data.shape}")
    print(f"  Prediction shape: {pred_data.shape}")
    print(f"  Metadata original shape: {metadata['original_spatial_shape']}")
    print(f"  Shapes match: {original_data.shape == pred_data.shape}")
    
    # Check affines
    print(f"\nAffine comparison:")
    print(f"  Original affine:\n{original_affine}")
    print(f"  Prediction affine:\n{pred_affine}")
    print(f"  Metadata original affine:\n{np.array(metadata['original_affine'])}")
    
    # Check if affines are close
    affine_close = np.allclose(original_affine, pred_affine, rtol=1e-5, atol=1e-5)
    print(f"  Affines match: {affine_close}")
    
    if not affine_close:
        diff = np.abs(original_affine - pred_affine)
        print(f"  Max affine difference: {np.max(diff)}")
        print(f"  Affine difference matrix:\n{diff}")
    
    # Check spacing
    original_spacing = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
    pred_spacing = np.sqrt(np.sum(pred_affine[:3, :3]**2, axis=0))
    
    print(f"\nSpacing comparison:")
    print(f"  Original spacing: {original_spacing}")
    print(f"  Prediction spacing: {pred_spacing}")
    print(f"  Metadata original spacing: {metadata.get('original_spacing', 'N/A')}")
    print(f"  Spacings match: {np.allclose(original_spacing, pred_spacing)}")
    
    # Check origin
    original_origin = original_affine[:3, 3]
    pred_origin = pred_affine[:3, 3]
    
    print(f"\nOrigin comparison:")
    print(f"  Original origin: {original_origin}")
    print(f"  Prediction origin: {pred_origin}")
    print(f"  Metadata original origin: {metadata.get('original_origin', 'N/A')}")
    print(f"  Origins match: {np.allclose(original_origin, pred_origin)}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"ALIGNMENT VERIFICATION SUMMARY:")
    all_good = (
        original_data.shape == pred_data.shape and
        np.allclose(original_affine, pred_affine, rtol=1e-5, atol=1e-5)
    )
    if all_good:
        print("✓ PASSED: Prediction properly aligns with original image")
    else:
        print("✗ FAILED: Alignment issues detected")
        if original_data.shape != pred_data.shape:
            print("  - Shape mismatch")
        if not np.allclose(original_affine, pred_affine, rtol=1e-5, atol=1e-5):
            print("  - Affine matrix mismatch")
    print(f"{'='*50}\n")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(description="Verify alignment of predictions with original images")
    parser.add_argument("--original", type=str, required=True, help="Path to original NIfTI file")
    parser.add_argument("--prediction", type=str, required=True, help="Path to prediction NIfTI file")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    args = parser.parse_args()
    
    verify_alignment(args.original, args.prediction, args.metadata)


if __name__ == "__main__":
    main() 