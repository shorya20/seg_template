import nibabel as nib
import sys
import os

def print_nifti_info(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        img = nib.load(file_path)
        header = img.header

        print(f"--- Header Info for: {os.path.basename(file_path)} ---")
        print(f"Dimensions: {header.get_data_shape()}")
        print(f"Datatype: {header.get_data_dtype()} (code: {header['datatype']})")
        print(f"Voxel Spacing (Zooms): {header.get_zooms()}")
        
        slope = header['scl_slope']
        inter = header['scl_inter']
        print(f"Slope (scl_slope): {slope if slope is not None else 'Not set'}")
        print(f"Intercept (scl_inter): {inter if inter is not None else 'Not set'}")

        # Attempt to get data and its range if slope/inter are not obviously 1/0
        # For large files, loading full data is slow, so this is a simplified check.
        # A full intensity range check would require loading data: img.get_fdata()
        # For now, we focus on datatype and slope/inter which guide HU conversion.

        # cal_min and cal_max are supposed to store the min/max of scaled data
        cal_min = header['cal_min']
        cal_max = header['cal_max']
        print(f"cal_min: {cal_min}")
        print(f"cal_max: {cal_max}")
        print("-----------------------------------------------------")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_check = sys.argv[1]
        # Try to construct path relative to a potential base ATM22 directory structure
        # This makes it easier if just the filename is passed
        # Assumes script is run from seg_template root
        potential_paths = [
            file_to_check, # if full path is given
            os.path.join("ATM22", "TrainBatch1", "imagesTr", file_to_check),
            os.path.join("ATM22", "TrainBatch2", "imagesTr", file_to_check),
            os.path.join("ATM22", "imagesVal", file_to_check) # If it's a validation file
        ]
        
        found_path = None
        for p in potential_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            print_nifti_info(found_path)
        else:
            print(f"Could not find {file_to_check} in common locations. Please provide a full or relative path from project root.")
            print(f"Searched: {potential_paths}")

    else:
        print("Usage: python data_prep/check_nifti_header.py <filename_or_path>")
        print("Example: python data_prep/check_nifti_header.py ATM_001_0000.nii.gz")
        print("Example: python data_prep/check_nifti_header.py ATM22/TrainBatch1/imagesTr/ATM_001_0000.nii.gz")

