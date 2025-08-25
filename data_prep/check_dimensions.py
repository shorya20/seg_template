import nibabel as nib
import numpy as np
import os

def check_file_shapes_and_sums(base_path, file_id):
    print(f"Checking dimensions and sums for {file_id} based on path: {base_path}")

    nifti_image_path_train_batch_1 = os.path.join(base_path, "ATM22", "TrainBatch1", "imagesTr", f"{file_id}_0000.nii.gz")
    nifti_label_path_train_batch_1 = os.path.join(base_path, "ATM22", "TrainBatch1", "labelsTr", f"{file_id}_0000.nii.gz")
    nifti_image_path_train_batch_2 = os.path.join(base_path, "ATM22", "TrainBatch2", "imagesTr", f"{file_id}_0000.nii.gz")
    nifti_label_path_train_batch_2 = os.path.join(base_path, "ATM22", "TrainBatch2", "labelsTr", f"{file_id}_0000.nii.gz")
    
    npy_image_path = os.path.join(base_path, "ATM22", "npy_files", "imagesTr", f"{file_id}_0000.npy")
    npy_label_path = os.path.join(base_path, "ATM22", "npy_files", "labelsTr", f"{file_id}_0000.npy")
    npy_lung_mask_path = os.path.join(base_path, "ATM22", "npy_files", "lungsTr", f"{file_id}_0000.npy") # Path for lung mask

    print(f"  Attempting NIfTI image (Batch1): {nifti_image_path_train_batch_1}")
    print(f"  Attempting NIfTI label (Batch1): {nifti_label_path_train_batch_1}")
    print(f"  Attempting NIfTI image (Batch2): {nifti_image_path_train_batch_2}")
    print(f"  Attempting NIfTI label (Batch2): {nifti_label_path_train_batch_2}")
    print(f"  Attempting NumPy image: {npy_image_path}")
    print(f"  Attempting NumPy label: {npy_label_path}")
    print(f"  Attempting NumPy lung mask: {npy_lung_mask_path}")

    nifti_image_path = None
    if os.path.exists(nifti_image_path_train_batch_1):
        print(f"  Found NIfTI image (Batch1): {nifti_image_path_train_batch_1}")
        nifti_image_path = nifti_image_path_train_batch_1
    elif os.path.exists(nifti_image_path_train_batch_2):
        print(f"  Found NIfTI image (Batch2): {nifti_image_path_train_batch_2}")
        nifti_image_path = nifti_image_path_train_batch_2
    else:
        print(f"  NIfTI image not found for {file_id} in TrainBatch1 or TrainBatch2.")

    nifti_label_path = None
    raw_label_sum = None
    if os.path.exists(nifti_label_path_train_batch_1):
        print(f"  Found NIfTI label (Batch1): {nifti_label_path_train_batch_1}")
        nifti_label_path = nifti_label_path_train_batch_1
    elif os.path.exists(nifti_label_path_train_batch_2):
        print(f"  Found NIfTI label (Batch2): {nifti_label_path_train_batch_2}")
        nifti_label_path = nifti_label_path_train_batch_2
    else:
        print(f"  NIfTI label not found for {file_id} in TrainBatch1 or TrainBatch2.")

    if nifti_image_path:
        try:
            img_nii = nib.load(nifti_image_path)
            print(f"  NIfTI Image ({nifti_image_path}): Shape = {img_nii.shape}")
        except Exception as e:
            print(f"  Error loading NIfTI Image {nifti_image_path}: {e}")
    
    if nifti_label_path:
        try:
            lbl_nii_obj = nib.load(nifti_label_path)
            print(f"  NIfTI Label ({nifti_label_path}): Shape = {lbl_nii_obj.shape}")
            lbl_nii_data = lbl_nii_obj.get_fdata()
            raw_label_sum = np.sum(lbl_nii_data)
            print(f"  NIfTI Label ({nifti_label_path}): Sum = {raw_label_sum}")
        except Exception as e:
            print(f"  Error loading or summing NIfTI Label {nifti_label_path}: {e}")

    if os.path.exists(npy_image_path):
        print(f"  Found NumPy image: {npy_image_path}")
        try:
            img_npy = np.load(npy_image_path, mmap_mode='r')
            print(f"  NumPy Image ({npy_image_path}): Shape = {img_npy.shape}")
        except Exception as e:
            print(f"  Error loading NumPy Image {npy_image_path}: {e}")
    else:
        print(f"  NumPy image not found: {npy_image_path}")

    npy_label_sum = None
    if os.path.exists(npy_label_path):
        print(f"  Found NumPy label: {npy_label_path}")
        try:
            lbl_npy = np.load(npy_label_path, mmap_mode='r')
            print(f"  NumPy Label ({npy_label_path}): Shape = {lbl_npy.shape}")
            npy_label_sum = np.sum(lbl_npy)
            print(f"  NumPy Label ({npy_label_path}): Sum = {npy_label_sum}")
        except Exception as e:
            print(f"  Error loading or summing NumPy Label {npy_label_path}: {e}")
    else:
        print(f"  NumPy label not found: {npy_label_path}")

    npy_lung_mask_sum = None
    if os.path.exists(npy_lung_mask_path):
        print(f"  Found NumPy lung mask: {npy_lung_mask_path}")
        try:
            lung_mask_npy = np.load(npy_lung_mask_path, mmap_mode='r')
            print(f"  NumPy Lung Mask ({npy_lung_mask_path}): Shape = {lung_mask_npy.shape}")
            npy_lung_mask_sum = np.sum(lung_mask_npy)
            print(f"  NumPy Lung Mask ({npy_lung_mask_path}): Sum = {npy_lung_mask_sum}")
        except Exception as e:
            print(f"  Error loading or summing NumPy Lung Mask {npy_lung_mask_path}: {e}")
    else:
        print(f"  NumPy lung mask not found: {npy_lung_mask_path}")

    print("-" * 30)

if __name__ == "__main__":
    project_base_path = os.getcwd()
    print(f"Script running from: {project_base_path}")
    print(f"Checking for base data directory: {os.path.join(project_base_path, 'ATM22')}")
    if not os.path.isdir(os.path.join(project_base_path, "ATM22")):
        print("ERROR: ATM22 directory not found at the expected location.")
        print("Please ensure you are running this script from the root of the 'seg_template' project directory,")
        print("and the ATM22 dataset is structured correctly within it.")
        exit()

    # Focus on the file mentioned in the logs
    files_to_check = ["ATM_250"] 
    
    for file_id in files_to_check:
        check_file_shapes_and_sums(project_base_path, file_id) # Renamed function
 