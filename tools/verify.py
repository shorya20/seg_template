# # --- START OF FILE debug_conversion.py ---
# import SimpleITK as sitk
# import numpy as np
# import sys
# import os

# # --- IMPORTANT: SET THIS PATH ---
# # Point this to one of the ORIGINAL NIFTI files that is causing problems.
# # Make sure this path is correct for your system.
# # Let's use ATM_508_0000.nii.gz as our test case.
# nifti_file_path = './ATM22/TrainBatch2/imagesTr/ATM_508_0000.nii.gz'
# # -----------------------------

# print(f"--- Debugging NIfTI file: {nifti_file_path} ---")

# if not os.path.exists(nifti_file_path):
#     print(f"ERROR: The file does not exist at the specified path: {nifti_file_path}")
#     print("Please make sure the path in the script is correct.")
#     sys.exit(1)

# try:
#     # Step 1: Read the image in its NATIVE format, without any type casting.
#     img_native = sitk.ReadImage(nifti_file_path)
#     print(f"\nStep 1: Successfully read the image.")
#     print(f"   - Original Pixel Type: {img_native.GetPixelIDTypeAsString()}")

#     # Step 2: Check for the existence of metadata keys.
#     has_slope = img_native.HasMetaDataKey("RescaleSlope")
#     has_intercept = img_native.HasMetaDataKey("RescaleIntercept")
#     print(f"\nStep 2: Checking for metadata keys...")
#     print(f"   - Has 'RescaleSlope' key:     {has_slope}")
#     print(f"   - Has 'RescaleIntercept' key: {has_intercept}")

#     # Step 3: If keys exist, print their values.
#     if has_slope and has_intercept:
#         slope = img_native.GetMetaData("RescaleSlope")
#         intercept = img_native.GetMetaData("RescaleIntercept")
#         print(f"\nStep 3: Found metadata values:")
#         print(f"   - Slope Value:     '{slope}' (type: {type(slope)})")
#         print(f"   - Intercept Value: '{intercept}' (type: {type(intercept)})")

#         # Step 4: Perform the conversion EXACTLY as in the main script.
#         print("\nStep 4: Attempting HU conversion...")
#         slope_f = float(slope)
#         intercept_f = float(intercept)
        
#         # This is the core logic
#         img_float = sitk.Cast(img_native, sitk.sitkFloat64)
#         converted_img = img_float * slope_f + intercept_f
#         print("   - Conversion calculation completed.")

#         # Step 5: Get statistics of the NEWLY CONVERTED image.
#         stats_array = sitk.GetArrayFromImage(converted_img)
#         print(f"\nStep 5: Statistics of the CONVERTED numpy array:")
#         print(f"   - Min value: {stats_array.min()}")
#         print(f"   - Max value: {stats_array.max()}")
#         print(f"   - Mean value: {stats_array.mean()}")
#         print(f"   - Voxels < -400: {np.sum(stats_array < -400)}")

#     else:
#         print("\nStep 3 & 4: Metadata for HU conversion not found. Skipping conversion.")
#         # Get statistics of the ORIGINAL image if no conversion happened.
#         stats_array = sitk.GetArrayFromImage(img_native)
#         print(f"\nStep 5: Statistics of the ORIGINAL numpy array (no conversion):")
#         print(f"   - Min value: {stats_array.min()}")
#         print(f"   - Max value: {stats_array.max()}")

# except Exception as e:
#     print(f"\nAN ERROR OCCURRED: {e}")
#    sys.exit(1) 

print("\n--- Debugging complete ---")
# --- END OF FILE debug_conversion.py ---
import numpy as np
import os
vol = np.load('/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/imagesTr/ATM_011_0000.npy')
msk = np.load('/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/lungsTr/ATM_011_0000.npy') > 0
body = vol > np.percentile(vol, 35)
print(body.sum(), msk.sum())
ratio_outside = (msk & ~body).sum() / max(1,(msk.sum()))
img_path = '/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/imagesTr/ATM_011_0000.npy'
print('outside_ratio for file {} = {}'.format(os.path.basename(img_path), ratio_outside))  # ~1.0 ⇒ ring; <<1.0 ⇒ mostly inside body
# import numpy as np
# from skimage.segmentation import clear_border
# from scipy.ndimage import binary_fill_holes, label, generate_binary_structure

# p = "/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/lungsTr/ATM_001_0000.npy"
# m = np.load(p) > 0
# # remove border-touching components slice-wise
# m = np.stack([clear_border(s) for s in m], axis=0)
# # 3D fill + keep two largest
# m = binary_fill_holes(m)
# lbl, n = label(m, structure=generate_binary_structure(3,2))
# if n:
#     b = np.bincount(lbl.ravel()); b[0] = 0
#     keep = b.argsort()[-2:]
#     m = np.isin(lbl, keep)
# np.save(p, m.astype(np.uint8))
# print("fixed mask saved:", p)


import os
import glob
import nibabel as nib
import numpy as np

# # Set the root directory
# root = '/vol/bitbucket/ss2424/project/seg_template/ATM22'

# # Find the file paths
# paths = glob.glob(root + '/TrainBatch*/imagesTr/ATM_001_0000.nii.gz') + glob.glob(root + '/imagesVal/ATM_001_0000.nii.gz')

# print('found', paths)

# # Check if any paths were found
# if paths:
#     f = paths[0]
#     img = nib.load(f)
#     hdr = img.header
#     slope, inter = hdr.get_slope_inter()
#     data = img.get_fdata()

#     print('slope,inter', slope, inter, 'dtype', data.dtype, 'min', float(np.nanmin(data)), 'max', float(np.nanmax(data)))
#     print('aff', img.affine)
import numpy as np
im=np.load('ATM22/npy_files/imagesTr/ATM_243_0000.npy')
lm=np.load('ATM22/npy_files/lungsTr/ATM_243_0000.npy')
print(im.shape,lm.shape,(im.shape==lm.shape))