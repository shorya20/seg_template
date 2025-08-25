import SimpleITK as sitk, numpy as np, glob, os

# nifti  = '/vol/bitbucket/ss2424/project/seg_template/ATM22/TrainBatch1/labelsTr/ATM_159_0000.nii.gz'
# npy    = '/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/labelsTr/ATM_159_0000.npy'

# def count_fg(files, loader):
#     return {os.path.basename(f): (loader(f) > 0).sum() for f in files}

# fg_raw  = count_fg(glob.glob(nifti),  lambda f: sitk.GetArrayFromImage(sitk.ReadImage(f)))
# fg_npy  = count_fg(glob.glob(npy),    lambda f: np.load(f, mmap_mode='r'))

# delta = {k: fg_npy.get(k.replace('.nii.gz','.npy'), 0)/v for k,v in fg_raw.items()}
# print("Median % of foreground kept:", np.median(list(delta.values())))


import numpy as np, matplotlib.pyplot as plt, seaborn as sns
arr = np.load("./ATM22/npy_files/imagesTr/ATM_508_0000.npy", mmap_mode="r")

print("min / max:", arr.min(), arr.max())
print("voxels >= -1000:", np.sum(arr >= -1000))

# sns.histplot(arr[arr>-1200].ravel()[::20])   # quick HU histogram
# plt.yscale("log")
# plt.show()

mask_path = "/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files/lungsTr/ATM_508_0000.npy"
mask = np.load(mask_path)
print(mask.shape)
print(mask.min(), mask.max())
print(mask.sum())
print(np.sum(mask > 0))