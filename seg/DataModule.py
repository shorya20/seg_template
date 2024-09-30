import pytorch_lightning as pl
import numpy as np
from glob import glob
import os
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Rotate90d,
    AsDiscreted,
    ResizeWithPadOrCropd, RandCropByPosNegLabeld, EnsureChannelFirstd,
)
from monai.data import Dataset, DataLoader, CacheDataset
import random
from aseg.utils import get_folder_names

# pl.seed_everything(123, workers=True)
seed_num = 123 # 321, 451, 275, 999
random.seed(seed_num)
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_path, dataset="ATM22", batch_size=1, test_batch_size=1,
                 train_val_ratio=0.85, augment=False, params=None, compute_dtype=torch.float32, crop_handle=False):
        """
            Goal:
            Args:

            Output:
        """
        super().__init__()
        parent_directory = os.path.dirname(data_path.rstrip('/'))
        img_file_name, gt_file_name, lungs_file_name = get_folder_names(dataset.split("_")[0])
        self.data_path = os.path.join(data_path, img_file_name)
        self.label_path = os.path.join(data_path, gt_file_name)
        self.lung_path = os.path.join(data_path, "C2_lbl")
        self.off_val_data_path = os.path.join(parent_directory, 'val/')
        self.off_val_lung_path = os.path.join(parent_directory, 'val_lungs/')
        if crop_handle:
            self.lung_path = os.path.join(data_path, 'crop_handle/') # switch to "lungs/" if do not wanna the gt+lungs
        self.weight_path = os.path.join(data_path, params['WEIGHT_DIR']) # weight_maps_edges if based on edges, weights if skeleton
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_val_ratio = train_val_ratio
        self.augment = augment
        self.train_transform = None
        self.off_val_transform = None
        self.val_transform = None
        self.test_transform = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.params = params
        self.one_hot = 2 if self.params['ONE_HOT'] else None # replace the first None with 2 if necessary
        self.compute_dtype = compute_dtype

    def get_data_paths(self):
        """
            Goal:
            Args:

            Output:
        """
        print(f"The self.data_path is {self.data_path}")
        print(f"The self.label_path is {self.label_path}")
        print(f"The self.lung_path is {self.lung_path}")
        print(f"The self.weight_path is {self.weight_path}")
        TRAIN_IMG_DIR = sorted(glob(os.path.join(self.data_path, '*.npy')))
        TRAIN_MASK_DIR = sorted(glob(os.path.join(self.label_path, '*.npy')))
        TRAIN_LUNGS_DIR = sorted(glob(os.path.join(self.lung_path, '*.npy')))
        TRAIN_WEIGHTS_DIR = sorted(glob(os.path.join(self.weight_path, '*.npy')))

        print(f"The length of TRAIN_IMG_DIR: {len(TRAIN_IMG_DIR)}")
        print(f"The length of TRAIN_MASK_DIR: {len(TRAIN_MASK_DIR)}")
        print(f"The length of TRAIN_LUNGS_DIR: {len(TRAIN_LUNGS_DIR)}")
        print(f"The length of TRAIN_WEIGHTS_DIR: {len(TRAIN_WEIGHTS_DIR)}")
        idx = random.sample(range(0, len(TRAIN_IMG_DIR)), self.params['MAX_CARDINALITY'])
        TRAIN_IMG_DIR = [TRAIN_IMG_DIR[index] for index in idx]
        TRAIN_MASK_DIR = [TRAIN_MASK_DIR[index] for index in idx]
        TRAIN_LUNGS_DIR = [TRAIN_LUNGS_DIR[index] for index in idx]
        TRAIN_WEIGHTS_DIR = [TRAIN_WEIGHTS_DIR[index] for index in idx]
        num_subjects = len(TRAIN_IMG_DIR)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        full_idxs = random.sample(range(0, num_subjects), num_subjects)
        train_idxs = random.sample(range(0, num_subjects), num_train_subjects)
        val_idxs = [i for i in full_idxs if i not in train_idxs]
        print(f"{len(full_idxs)} {len(train_idxs)} {len(val_idxs)}")

        OFF_VAL_IMG_DIR = sorted(glob(os.path.join(self.off_val_data_path, '*.nii.gz')))
        OFF_VAL_LUNGS_DIR = sorted(glob(os.path.join(self.off_val_lung_path, "*.nii.gz")))

        TEST_IMG_DIR = sorted(glob(os.path.join(self.off_val_data_path, '*.nii.gz')))
        TEST_LUNGS_DIR = sorted(glob(os.path.join(self.off_val_lung_path, "*.nii.gz")))

        return list(np.array(TRAIN_IMG_DIR)[train_idxs]), \
               list(np.array(TRAIN_LUNGS_DIR)[train_idxs]), \
               list(np.array(TRAIN_MASK_DIR)[train_idxs]), \
               list(np.array(TRAIN_WEIGHTS_DIR)[train_idxs]), \
               list(np.array(TRAIN_IMG_DIR)[val_idxs]), \
               list(np.array(TRAIN_LUNGS_DIR)[val_idxs]), \
               list(np.array(TRAIN_MASK_DIR)[val_idxs]), \
               list(np.array(TRAIN_WEIGHTS_DIR)[val_idxs]), \
               OFF_VAL_IMG_DIR, OFF_VAL_LUNGS_DIR, TEST_IMG_DIR, TEST_LUNGS_DIR

    def prepare_data_monai(self) -> None:
        """
            Goal:
            Args:

            Output:
        """
        data_training_path, lungs_training_path, label_training_path, weights_training_path, \
        data_val_path, lungs_val_path, label_val_path, weight_val_path, \
        img_off_val_path, lungs_off_val_path, img_test_path, lungs_test_path = self.get_data_paths()
        self.train_files = [{'image': image, 'lung': lung, 'label': label, 'weight': weight} for
                            image, lung, label, weight in zip(data_training_path, lungs_training_path, label_training_path, weights_training_path)]
        self.val_files = [{'image': image, 'lung': lung, 'label': label, 'weight': weight} for
                          image, lung, label, weight in zip(data_val_path, lungs_val_path, label_val_path, weight_val_path)]
        self.off_val_files = [{'image': image, 'lung': lung} for
                          image, lung in zip(img_off_val_path, lungs_off_val_path)]
        self.test_files = [{'image': image, 'lung': lung} for
                          image, lung in zip(img_test_path, lungs_test_path)]
        print(f"Number of Patients in training set {len(self.train_files)}")
        print(f"Number of Patients in validation set {len(self.val_files)}")
        print(f"Number of Patients in official validation set {len(self.off_val_files)}")
        print(f"Number of Patients in official test set {len(self.test_files)}")
      
    def get_preprocess_transforms(self):
        """
            Goal:
            Args:

            Output:
        """
        train_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung', 'label', 'weight']),
                EnsureChannelFirstd(keys=['image', 'lung', 'label', 'weight'], strict_check=True),
                CropForegroundd(keys=['image', 'lung', 'label', 'weight'], source_key='lung',
                                margin=[1,1,1], allow_smaller=True),
                RandCropByPosNegLabeld(keys=['image', 'lung', 'label', 'weight'],
                                   label_key='label',
                                   spatial_size=self.params['PATCH_SIZE'],
                                   num_samples=1,
                                   pos=9,
                                   neg=0,
                                   allow_smaller=True),
                ResizeWithPadOrCropd(keys=['image', 'lung', 'label', 'weight'], spatial_size=self.params['PATCH_SIZE']),
                AsDiscreted(keys=['label'], to_onehot=self.one_hot),
                ToTensord(keys=['image', 'lung', 'label', 'weight'], dtype=self.compute_dtype),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung', 'label', 'weight'], image_only=False),
                EnsureChannelFirstd(keys=['image', 'lung', 'label', 'weight']),
                CropForegroundd(keys=['image', 'lung', 'label', 'weight'], source_key='lung',
                                margin=[1,1,1], allow_smaller=True),
                AsDiscreted(keys=['label'], to_onehot=self.one_hot),
                ToTensord(keys=['image', 'lung', 'label', 'weight'], dtype=self.compute_dtype),
            ]
        )
        aff_traning_val_transform = Compose(
            [
                LoadImaged(keys=['image', 'lung', 'label', 'weight'], image_only=False),
                EnsureChannelFirstd(keys=['image', 'lung', 'label', 'weight']),
                CropForegroundd(keys=['image', 'lung', 'label', 'weight'], source_key='lung',
                                margin=[1, 1, 25], allow_smaller=True),
                AsDiscreted(keys=['label'], to_onehot=self.one_hot),
                ToTensord(keys=['image', 'lung', 'label', 'weight'], dtype=self.compute_dtype),
            ]
        )
        off_val_transform = Compose(
            [
                LoadImaged(keys=['image', 'lung'], image_only=False),
                EnsureChannelFirstd(keys=['image', 'lung']),
                CropForegroundd(keys=['image', 'lung'], source_key='lung',
                                margin=[1,1,50], allow_smaller=True),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                Rotate90d(keys=['image', 'lung'], k=3),
                ToTensord(keys=['image', 'lung'], dtype=self.compute_dtype),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung'], image_only=False),
                EnsureChannelFirstd(keys=['image', 'lung']),
                CropForegroundd(keys=['image', 'lung'], source_key='lung',
                                margin=[1,1,50], allow_smaller=True),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                Rotate90d(keys=['image', 'lung'], k=3),
                ToTensord(keys=['image', 'lung'], dtype=self.compute_dtype),
            ]
        )
        return train_transforms, val_transforms, aff_traning_val_transform, off_val_transform, test_transforms

    def setup(self, stage=None, dataset_library='monai') -> None:
        """
            Goal:
            Args:

            Output:
        """
        self.train_preprocess, self.val_preprocess, self.aff_traning_val_preprocess, self.off_val_preprocess, self.test_preprocess = self.get_preprocess_transforms()

        self.train_transform = self.train_preprocess if not self.augment else None # TODO:augmentation should b e implemented here
        self.val_transform = self.val_preprocess if not self.augment else None # TODO:augmentation should b e implemented here
        self.aff_traning_val_transform = self.aff_traning_val_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here
        self.off_val_transform = self.off_val_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here
        self.test_transform = self.test_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here

        if self.params['CACHE_DS']:
            self.train_set = CacheDataset(data=self.train_files, transform=self.train_transform)
            self.val_set = CacheDataset(data=self.val_files, transform=self.val_transform)
        else:
            self.train_set = Dataset(data=self.train_files, transform=self.train_transform)
            self.val_set = Dataset(data=self.val_files, transform=self.val_transform)
        self.aff_training_val_set = Dataset(data=self.val_files, transform=self.aff_traning_val_transform)
        self.off_val_set = Dataset(data=self.off_val_files, transform=self.off_val_transform)
        self.test_set = Dataset(data=self.test_files, transform=self.test_transform)

        # print(self.train_set.__getitem__(1))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_set,
                          batch_size=self.params['BATCH_SIZE'],
                          pin_memory=self.params['PIN_MEMORY'],
                          num_workers=self.params['NUM_WORKERS'],)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_set,
                          batch_size=1,
                          num_workers=self.params['NUM_WORKERS']//2)

    def aff_training_val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.aff_training_val_set,
                          batch_size=1,
                          num_workers=1,)

    def off_val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.off_val_set,
                          batch_size=1,
                          num_workers=1,)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_set,
                          batch_size=1,
                          num_workers=1,)

