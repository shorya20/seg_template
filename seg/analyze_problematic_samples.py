#!/usr/bin/env python
"""
Script to analyze problematic validation samples that show IoU=0 or very low IoU.
This helps identify if these samples have data quality issues or unusual characteristics.
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def analyze_sample(npy_path, label_path=None, lung_path=None):
    """Analyze a single sample for potential issues."""
    results = {}
    
    # Load the image
    img_data = np.load(npy_path)
    results['shape'] = img_data.shape
    results['dtype'] = str(img_data.dtype)
    results['min_value'] = float(img_data.min())
    results['max_value'] = float(img_data.max())
    results['mean_value'] = float(img_data.mean())
    results['std_value'] = float(img_data.std())
    
    # Check for NaN or Inf values
    results['has_nan'] = bool(np.isnan(img_data).any())
    results['has_inf'] = bool(np.isinf(img_data).any())
    
    # Load and analyze label if available
    if label_path and Path(label_path).exists():
        label_data = np.load(label_path)
        results['label_shape'] = label_data.shape
        results['label_sum'] = int(label_data.sum())
        results['label_volume_voxels'] = int((label_data > 0).sum())
        
        # Check connectivity of label
        labeled_array, num_components = ndimage.label(label_data > 0)
        results['label_num_components'] = num_components
        
        # Get component sizes
        component_sizes = []
        for i in range(1, num_components + 1):
            component_sizes.append(int((labeled_array == i).sum()))
        results['label_component_sizes'] = sorted(component_sizes, reverse=True)[:5]  # Top 5 largest
        
        # Check if label is at image boundary
        boundary_mask = np.zeros_like(label_data, dtype=bool)
        boundary_mask[0, :, :] = True
        boundary_mask[-1, :, :] = True
        boundary_mask[:, 0, :] = True
        boundary_mask[:, -1, :] = True
        boundary_mask[:, :, 0] = True
        boundary_mask[:, :, -1] = True
        results['label_touches_boundary'] = bool((label_data > 0 & boundary_mask).any())
        
        # Calculate bounding box of label
        nonzero_coords = np.where(label_data > 0)
        if len(nonzero_coords[0]) > 0:
            results['label_bbox'] = {
                'z_range': [int(nonzero_coords[0].min()), int(nonzero_coords[0].max())],
                'y_range': [int(nonzero_coords[1].min()), int(nonzero_coords[1].max())],
                'x_range': [int(nonzero_coords[2].min()), int(nonzero_coords[2].max())]
            }
        else:
            results['label_bbox'] = None
    
    # Load and analyze lung mask if available
    if lung_path and Path(lung_path).exists():
        lung_data = np.load(lung_path)
        results['lung_volume_voxels'] = int((lung_data > 0).sum())
        
        # Check if label is within lung mask
        if label_path and Path(label_path).exists():
            overlap = (label_data > 0) & (lung_data > 0)
            results['label_lung_overlap_ratio'] = float(overlap.sum() / max((label_data > 0).sum(), 1))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze problematic validation samples')
    parser.add_argument('sample_list', help='Text file with list of problematic samples or single .npy file')
    parser.add_argument('--data_dir', default='./ATM22/npy_files', help='Base directory for data files')
    parser.add_argument('--output', help='Output JSON file for analysis results')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Determine if input is a list file or single sample
    sample_paths = []
    if args.sample_list.endswith('.txt'):
        with open(args.sample_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_paths.append(line)
    else:
        sample_paths = [args.sample_list]
    
    # Analyze each sample
    all_results = {}
    for sample_path in sample_paths:
        # Extract sample name
        if '/' in sample_path:
            sample_name = Path(sample_path).stem
        else:
            sample_name = sample_path.replace('.npy', '')
        
        print(f"\nAnalyzing {sample_name}...")
        
        # Construct paths
        data_dir = Path(args.data_dir)
        if sample_path.endswith('.npy'):
            img_path = sample_path
        else:
            img_path = data_dir / 'imagesTr' / f'{sample_name}.npy'
        
        label_path = data_dir / 'labelsTr' / f'{Path(img_path).name}'
        lung_path = data_dir / 'lungsTr' / f'{Path(img_path).name}'
        
        # Analyze
        try:
            results = analyze_sample(img_path, label_path, lung_path)
            all_results[sample_name] = results
            
            # Print summary
            print(f"  Shape: {results['shape']}")
            print(f"  Value range: [{results['min_value']:.2f}, {results['max_value']:.2f}]")
            if 'label_volume_voxels' in results:
                print(f"  Label volume: {results['label_volume_voxels']} voxels")
                print(f"  Label components: {results['label_num_components']}")
                if results['label_touches_boundary']:
                    print("  WARNING: Label touches image boundary!")
            if 'label_lung_overlap_ratio' in results:
                print(f"  Label-lung overlap: {results['label_lung_overlap_ratio']*100:.1f}%")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[sample_name] = {'error': str(e)}
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Create summary statistics
    print("\n=== SUMMARY ===")
    total_samples = len(all_results)
    print(f"Total samples analyzed: {total_samples}")
    
    # Find common issues
    boundary_touching = sum(1 for r in all_results.values() 
                          if r.get('label_touches_boundary', False))
    multi_component = sum(1 for r in all_results.values() 
                         if r.get('label_num_components', 0) > 1)
    small_labels = sum(1 for r in all_results.values() 
                      if r.get('label_volume_voxels', 0) < 1000)
    
    if boundary_touching > 0:
        print(f"  - {boundary_touching}/{total_samples} samples have labels touching boundary")
    if multi_component > 0:
        print(f"  - {multi_component}/{total_samples} samples have multi-component labels")
    if small_labels > 0:
        print(f"  - {small_labels}/{total_samples} samples have small labels (<1000 voxels)")


if __name__ == '__main__':
    main() 