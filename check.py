# from pathlib import Path
# import random
# import numpy as np, glob
# f = random.choice(glob.glob("ATM22/npy_files/labelsTr/ATM_203_0000.npy"))
# lbl = np.load(f, mmap_mode="r")
# print(f, lbl.dtype, lbl.min(), lbl.max(), lbl.mean(), lbl.sum())


# import numpy as np, glob

# for p in glob.glob("ATM22/npy_files/labelsTr/ATM_203_0000.npy"):
#     a = np.load(p)
#     print(p, "foreground voxels =", int((a>0).sum()))

import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

# Paths for the specific validation file
img_path  = "/vol/bitbucket/ss2424/project/seg_template/ATM22/val/ATM_281_0000.npy"
mask_path = "/vol/bitbucket/ss2424/project/seg_template/ATM22/val_lungs/ATM_281_0000.npy"

print(f"--- Visual Check ---")
print(f"Image: {img_path}")
print(f"Mask:  {mask_path}")
print(f"--------------------")

# 1) Load the processed volume and the mask
try:
    vol = np.load(img_path)
    mask = np.load(mask_path)
except FileNotFoundError as e:
    print(f"Error: Could not find the file - {e}")
    print("Please ensure the validation data has been processed correctly.")
    sys.exit(1)

# NEW: Debug prints to inspect mask
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask min/max: {mask.min()} / {mask.max()}")
print(f"Mask sum: {mask.sum()}")
print(f"Non-zero voxels: {np.sum(mask > 0)}")

coords = np.column_stack(np.where(mask > 0))


# 2) Compute the non-zero bounding box of the mask
num_nonzero = np.sum(mask > 0)

if num_nonzero == 0:
    print("The lung mask is empty (no values >0), so an overlay cannot be generated.")
    sys.exit()

z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0)
print("Mask bounding box (z,y,x):", (z_min, z_max), (y_min, y_max), (x_min, x_max))

# 3) Pick a slice to visualize (e.g. the center of the mask in Z)
z_center = int((z_min + z_max) / 2)


# 4) Plot the axial slice with the mask overlaid
plt.figure(figsize=(10, 10))
plt.imshow(vol[z_center], cmap="gray", interpolation="none")

# contour the mask boundary in red
plt.contour(mask[z_center], levels=[0.5], colors="r", linewidths=1)


# Ensure the output directory exists
output_dir = "ATM22/figs"
os.makedirs(output_dir, exist_ok=True)

# Save the figure with the correct filename for the file being checked
output_filename = os.path.join(output_dir, "val_ATM_281_0000_overlay.png")

plt.title(f"Axial slice {z_center} with lung mask overlay for ATM_281_0000")
plt.axis("off")
plt.savefig(output_filename)
plt.show()

print(f"\nVisual check complete. Image saved to: {output_filename}")