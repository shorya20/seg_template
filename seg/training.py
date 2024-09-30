import os
import argparse
import json
import sys
from typing import Any, Optional
from monai.utils import first
import numpy as np
import time
import random
import datetime
import nibabel as nib
from utils import (find_file_in_path, get_model, get_loss, get_optimizer, get_metric, get_scheduler,
                   get_folder_names, save_val_2path, get_lcc, find_breakages, adjust_weights)
from DataModule import SegmentationDataModule
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    AsDiscrete,
    Rotate90,
)
from dotenv import load_dotenv, find_dotenv
from task1 import evaluation_branch_metrics
import csv
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, rlimit[1]))
torch.set_float32_matmul_precision('high')

pl.seed_everything(42, workers=True)

def parse_args():
    parser = argparse.ArgumentParser(description="The airway segmentation on AIIB23")

    # patch size - for now static
    parser.add_argument("--patch_size", type=int, default=int(os.getenv('PATCH_SIZE')), help="Minimum pixel value")
    parser.add_argument("--one_hot", action="store_true", help="run on one-hot encoding")
    parser.add_argument("--mixed", action="store_true", help="run with mixed precision training")

    # Voxel values - for now static
    parser.add_argument("--pixel_value_min", type=int, default=float(os.getenv('PIXEL_VALUE_MIN')), help="Minimum pixel value")
    parser.add_argument("--pixel_value_max", type=int, default=float(os.getenv('PIXEL_VALUE_MAX')), help="Maximum pixel value")
    parser.add_argument("--pixel_norm_min", type=float, default=float(os.getenv('PIXEL_NORM_MIN')), help="Minimum normalized pixel value")
    parser.add_argument("--pixel_norm_max", type=float, default=float(os.getenv('PIXEL_NORM_MAX')), help="Maximum normalized pixel value")
    parser.add_argument("--voxel_size", type=int, default=float(os.getenv('VOXEL_SIZE')), help="Voxel size")

    # Model Hyperparameters - static for the moment
    parser.add_argument("--model_name", type=str, default=os.getenv('MODEL_NAME'), help="Model name")
    parser.add_argument("--spatial_dims", type=int, default=int(os.getenv('SPATIAL_DIMS')), help="Spatial dimensions")
    parser.add_argument("--in_channels", type=int, default=int(os.getenv('IN_CHANNELS')), help="Input channels")
    parser.add_argument("--out_channels", type=int, default=int(os.getenv('OUT_CHANNELS')), help="Output channels")
    parser.add_argument("--n_layers", type=int, default=int(os.getenv('N_LAYERS')), help="Number of layers")
    parser.add_argument("--channels", type=int, default=int(os.getenv('CHANNELS')), help="Channels")
    parser.add_argument("--strides", type=int, default=int(os.getenv('STRIDES')), help="Strides")
    parser.add_argument("--num_res_units", type=int, default=int(os.getenv('NUM_RES_UNITS')), help="Number of residual units")
    parser.add_argument("--dropout", type=float, default=float(os.getenv('DROPOUT')), help="Dropout rate")
    parser.add_argument("--norm", type=str, default=os.getenv('NORM'), help="Normalization type")

    # Training Hyperparameters
    parser.add_argument("--loss_func", type=str, default=os.getenv('LOSS_FUNC'), help="The loss function to use")
    parser.add_argument("--alpha", type=float, default=float(os.getenv('ALPHA_VALUE')), help="The alpha value")
    parser.add_argument("--r_d", type=float, default=1., help="The decay pattern value")
    parser.add_argument("--r_l", type=float, default=0.7, help="The power in GUL!")
    parser.add_argument("--scheduler", type=str, default=os.getenv('SCHEDULER_TYPE'), help="Type of scheduler")
    parser.add_argument("--patience", type=int, default=int(os.getenv('PATIENCE')), help="The learning rate patience")
    parser.add_argument("--optimizer", type=str, default=os.getenv('OPTIMIZER'), help="The optimizer to use")
    parser.add_argument("--lr", type=float, default=os.getenv('LR'), help="The learning rate")
    parser.add_argument("--max_epochs", type=int, default=os.getenv('NUM_EPOCHS'), help="how many epochs?")
    parser.add_argument("--max_cardinality", type=int, default=os.getenv('MAX_CARDINALITY'), help="size of dataset")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv('BATCH_SIZE')), help="size of dataset")
    parser.add_argument("--num_workers", type=int, default=int(os.getenv('NUM_WORKERS')), help="size of dataset")

    # File paths
    parser.add_argument("--exp_path", type=str, default="", help="The experiment path")
    parser.add_argument("--weights_path", type=str, default="", help="The best model weight path")
    parser.add_argument("--dataset", type=str, default="AIIB23", help="The DATASET to use")
    parser.add_argument("--weights_dir", type=str, default="Wr1alpha0.2", help="The weight map of mgul path")

    # Adaptive weight change AWC
    parser.add_argument("--awc", action="store_true", default=False, help="activate AWC or not")
    parser.add_argument("--awc_factor", type=float, default=1.2, help="The learning rate")

    # Other options
    parser.add_argument("--amp", action="store_true", default=False, help="For improving the training")
    parser.add_argument("--cache_ds", action="store_true", default=False, help="Caching dataset or not")
    parser.add_argument("--save_val", action="store_true", default=False, help="Caching dataset or not")

    # Phase of the experiment
    parser.add_argument("--training", action="store_true", help="Run training phase")
    parser.add_argument("--validation", action="store_true", help="Run validation phase")
    parser.add_argument("--testing", action="store_true", help="Run testing phase")
    parser.add_argument("--awc_val", action="store_true", help="Run awc validation phase")

    args = parser.parse_args()
    args.use_pretrained = args.weights_path != ""
    args.one_hot = True if (args.loss_func == "DiceLoss" or args.loss_func == "TverskyLoss") else False
    args.out_channels = 2 if args.one_hot else 1
    # the following for type 1 is not important but for type 2 is the weight maps dir
    args.weights_dir = f"Wr{str(args.r_d)}alpha{str(args.alpha)}" if args.loss_func == "GUL" else f"WBr{str(args.r_d)}alpha{str(args.alpha)}"
    # args.weights_dir = f"WCBr{str(args.r_d)}alpha{str(args.alpha)}" if args.loss_func == "CBEL" else args.weights_dir
    args.weights_dir = f"WBr{str(args.r_d)}alpha{str(args.alpha)}" if args.loss_func == "CBEL" else args.weights_dir
    args.max_cardinality = 120 if args.dataset == "AIIB23" else 299
    if args.dataset == "AeroPath":
        args.max_cardinality = 27

    return args

load_dotenv(find_dotenv("config.env"))

now = datetime.datetime.now()
day = now.day
hour = now.hour
month = now.month

set_determinism(seed=42)

# parser inputs
args = parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    # 'PATCH_SIZE': (args.patch_size, args.patch_size, args.patch_size),
    # 'PATCH_SIZE': (384, 256, 512),
# Parameters
params_dict = {
    'PATCH_SIZE': (args.patch_size, args.patch_size, args.patch_size),
    'BATCH_SIZE': args.batch_size,
    'MAX_CARDINALITY': args.max_cardinality,
    'NUM_WORKERS': int(os.getenv('NUM_WORKERS', args.num_workers)),
    'PIN_MEMORY': True,
    'PIXEL_VALUE_MIN': int(os.getenv('PIXEL_VALUE_MIN', args.pixel_value_min)),
    'PIXEL_VALUE_MAX': int(os.getenv('PIXEL_VALUE_MAX', args.pixel_value_max)),
    'PIXEL_NORM_MIN': float(os.getenv('PIXEL_NORM_MIN', args.pixel_norm_min)),
    'PIXEL_NORM_MAX': float(os.getenv('PIXEL_NORM_MAX', args.pixel_norm_max)),
    'VOXEL_SIZE': (args.voxel_size, args.voxel_size, args.voxel_size),
    'NUM_EPOCHS': args.max_epochs,
    'CACHE_DS': bool(args.cache_ds) if args.training else False,
    'AVAILABLE_GPUs': torch.cuda.device_count(),
    'DEVICE_NO': int(os.getenv('DEVICE_NO')),
    'VALIDATION_PHASE': bool(args.validation),
    'TRAINING_PHASE': bool(args.training),
    'TEST_PHASE': bool(args.testing),
    'USE_PRETRAINED': bool(args.use_pretrained),
    'WEIGHT_DIR': args.weights_dir,
    'ONE_HOT': args.one_hot,
}

pin_memory = torch.cuda.is_available()
device = torch.device(f"cuda:{params_dict['DEVICE_NO']}" if torch.cuda.is_available() else "cpu")

exp_path = f"{args.loss_func}_" \
           f"{args.dataset}_" \
           f"{month}_{day}_{hour}_" \
           f"{args.model_name}_" \
           f"prtr{args.use_pretrained}_" \
           f"{args.optimizer}_" \
           f"{str(args.patch_size)}_" \
           f"b{str(args.batch_size)}_" \
           f"p{str(args.patience)}_" \
           f"{args.lr}_" \
           f"alpha{args.alpha}_" \
           f"r_d{args.r_d}" \
           f"r_l{args.r_l}" \
           f"{args.scheduler}"

if params_dict['VALIDATION_PHASE'] or params_dict['TEST_PHASE']:
    exp_path = os.getenv('OUTPUT_PATH') + str(args.exp_path)
    best_model_path = exp_path + ("/model/" if not args.awc_val else "/model_AWC/") + str(args.weights_path)
    output_path = os.getenv('OUTPUT_PATH') + args.exp_path
else:
    output_path = os.getenv('OUTPUT_PATH') + exp_path

if params_dict['TRAINING_PHASE']:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + "/model/", exist_ok=True)
    os.makedirs(output_path + "/outputs/", exist_ok=True)
    os.makedirs(output_path + "/val_outputs/", exist_ok=True)
    os.makedirs(output_path + "/off_val_outputs/", exist_ok=True)
    os.makedirs(output_path + "/off_test_outputs/", exist_ok=True)
    os.makedirs(output_path + "/visualization/", exist_ok=True)
    os.makedirs(output_path + "/AWC_STEP/", exist_ok=True)
dataset_path = os.getenv('DATASET_PATH') + args.dataset
dataset_path = dataset_path + "/npy_files/"
print(f"The output path is {output_path}")
print(f"The dataset path is {dataset_path}")
print("parameters...")

for item in params_dict:
    print(f"{item} = {params_dict[item]}")

data_module = SegmentationDataModule(
    data_path=dataset_path,
    batch_size=args.batch_size,
    dataset=args.dataset,
    params=params_dict
)
data_module.prepare_data_monai()
data_module.setup()

# Save the arguments to a JSON file
with open(output_path+"/params.json", 'w') as json_file:
    json.dump(params_dict, json_file, indent=2)
csv_file = output_path + "/results.csv"


model_params = {
    'MODEL_NAME': args.model_name,
    'NORM': args.norm,
    'SPATIAL_DIMS': args.spatial_dims,
    'IN_CHANNELS': args.in_channels,
    'OUT_CHANNELS': args.out_channels,
    'N_LAYERS': args.n_layers,
    'CHANNELS': args.channels,
    'STRIDES': args.strides,
    'NUM_RES_UNITS': args.num_res_units,
    'DROPOUT': args.dropout,
    'USE_PRETRAINED': args.use_pretrained,
    'WEIGHTS_PATH': args.weights_path
}
# Save the arguments to a JSON file
with open(output_path+"/model_params.json", 'w') as json_file:
    json.dump(model_params, json_file, indent=2)


class SegmentationNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = get_model(params=model_params, device=device if args.validation else "cpu")
        self.loss_function = get_loss(l_name=args.loss_func, sigmoid=True, alpha=float(args.alpha), rl=args.r_l)
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_pred_save = Compose([AsDiscrete(argmax=True)])
        self.post_pred_sc = Compose([AsDiscrete(threshold=0.5)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
        self.meanDice_metric = get_metric(metric_name='DiceMetric', include_background=True, reduction="mean")
        self.meanIoU_metric = get_metric(metric_name='MeanIoU', include_background=True, reduction="mean")
        self.best_val_dice = 0
        self.best_val_iou = 0
        self.best_val_epoch = 0
        self.max_epochs = args.max_epochs
        self.check_val = 10
        self.warmup_epochs = 10
        self.dice_metric_values = []
        self.iou_metric_values = []
        self.epoch_loss_values = []
        self.step_training_loss = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def train_dataloader(self):
        return data_module.train_dataloader()

    def val_dataloader(self):
        return data_module.val_dataloader()

    def test_dataloader(self): # This is just for computing the full metrics on the validation set during the training
        return data_module.aff_training_val_dataloader()

    def off_val_dataloader(self):
        return data_module.off_val_dataloader()

    def off_test_dataloader(self):
        return data_module.test_dataloader()

    def configure_optimizers(self):
        optimizer = get_optimizer(m_params=self._model.parameters(),
                          opt_name=args.optimizer,
                          lr=args.lr if args.scheduler == 'ReduceLROnPlateau' else float(args.lr) * 10)
        scheduler = get_scheduler(optimizer=optimizer, type=args.scheduler,
                                  step_size=int(args.patience),
                                  patience=int(args.patience))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou",
                "frequency": 1,
            }
        }

    def on_train_start(self):
        self.training_start_time = time.time()

    def on_train_end(self):
        training_end_time = time.time()
        total_training_time = training_end_time - self.training_start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

    def training_step(self, batch, batch_idx):
        inputs, labels, lungs, weights = (
            batch["image"],
            batch["label"],
            batch["lung"],
            batch["weight"]
        )
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        tensorboard_logs = {'train_loss': loss.item()}
        d = {
            'loss': loss.item()
        }
        self.step_training_loss.append(d)
        self.log_dict(d, on_step=True, logger=True, prog_bar=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def on_train_epoch_end(self):
        avg_loss = sum([x["loss"] for x in self.step_training_loss]) / len(self.step_training_loss)
        self.epoch_loss_values.append(avg_loss)
        self.step_training_loss.clear()

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels, val_lungs, val_weights = (
            batch["image"],
            batch["label"],
            batch["lung"],
            batch["weight"]
        )
        outputs = sliding_window_inference(
                inputs=val_inputs, roi_size=args.patch_size, sw_batch_size=1,
                mode='constant', predictor=self.forward,
                overlap=0.25)
        if args.loss_func == 'GUL' or args.loss_func == 'BEL':
            loss = self.loss_function(outputs, val_labels, val_weights)
        elif args.loss_func == 'CBEL':
            loss = self.loss_function(outputs, val_labels, val_weights, val_lungs)
        else:
            loss = self.loss_function(outputs, val_labels)
        val_outputs = [self.post_pred(i) if args.one_hot else torch.sigmoid(i) for i in decollate_batch(outputs)]
        val_labels = [i for i in decollate_batch(val_labels)]
        self.meanDice_metric(val_outputs, val_labels)
        self.meanIoU_metric(val_outputs, val_labels)
        d = {
            'val_loss': loss,
            "val_number": len(val_outputs),
            "val_dice": self.meanDice_metric.aggregate().item(),
            "val_iou": self.meanIoU_metric.aggregate().item()
        }
        self.val_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.val_step_outputs:
            val_loss += output['val_loss'].sum().item()
            num_items += output['val_number']
        mean_val_dice = self.meanDice_metric.aggregate().item()
        mean_val_iou = self.meanIoU_metric.aggregate().item()
        self.meanDice_metric.reset()
        self.meanIoU_metric.reset()
        mean_val_loss = torch.tensor(val_loss/num_items)
        tensorboard_logs = {
            'val_loss': mean_val_loss,
            'val_dice': mean_val_dice,
            'val_iou': mean_val_iou
        }
        if mean_val_iou > self.best_val_iou:
            self.best_val_iou = mean_val_iou
            self.best_val_epoch = self.current_epoch

        print(
            f"\n"
            f"\nBest IoU: {self.best_val_iou} at epoch {self.best_val_epoch}, with LR {self.optimizers().param_groups[0]['lr']}\n"
        )
        self.dice_metric_values.append(mean_val_dice)
        self.iou_metric_values.append(mean_val_iou)
        self.val_step_outputs.clear()
        self.log_dict(tensorboard_logs, on_step=False, logger=True, prog_bar=True)
        return {'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        tst_inputs, tst_labels, tst_lungs, tst_weights = (
            batch["image"],
            batch["label"],
            batch["lung"],
            batch["weight"]
        )
        outputs = sliding_window_inference(
            inputs=tst_inputs, roi_size=args.patch_size, sw_batch_size=args.batch_size,
            mode='constant', predictor=self.forward,
            overlap=0.25)
        tst_outputs = [self.post_pred(i) if args.one_hot else self.post_pred_sc(torch.sigmoid(i)) for i in decollate_batch(outputs)]
        tst_labels = [i for i in decollate_batch(tst_labels)]
        self.meanDice_metric(tst_outputs, tst_labels)
        self.meanIoU_metric(tst_outputs, tst_labels)

        tst_filename = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
        spt_size = batch['image_meta_dict']['spatial_shape'][0].tolist()
        x1, x2, x3 = batch['foreground_start_coord'][0][0], batch['foreground_start_coord'][0][1], \
                batch['foreground_start_coord'][0][2]
        y1, y2, y3 = batch['foreground_end_coord'][0][0], batch['foreground_end_coord'][0][1], \
                batch['foreground_end_coord'][0][2]
        output_corr = np.zeros(spt_size)
        label_corr = np.zeros(spt_size)

        tst_outputs = [self.post_pred_save(item) if args.one_hot else item for item in tst_outputs]
        tst_labels = [self.post_pred_save(item) if args.one_hot else item for item in tst_labels]

        output_corr[x1:y1, x2:y2, x3:y3] = np.squeeze(tst_outputs[0].type(torch.int64).cpu().numpy())
        label_corr[x1:y1, x2:y2, x3:y3] = np.squeeze(tst_labels[0].type(torch.int64).cpu().numpy())

        iou, DLR, DBR, precision, leakages, amr, lcc = evaluation_branch_metrics(
            fid=tst_filename,
            label=label_corr,
            pred=output_corr
        )

        d = {
            "tst_filename": tst_filename,
            "tst_number": len(tst_outputs),
            "tst_dice": self.meanDice_metric.aggregate().item(),
            "tst_iou": self.meanIoU_metric.aggregate().item(),
            "tst_IoU": iou,
            "tst_DLR": DLR,
            "tst_DBR": DBR,
            "tst_Prec.": precision,
            "tst_Leak.": leakages,
            "tst_AMR.": amr,
        }

        for key, value in d.items():
            print(f"{key}: {value}", end=", ")

        if args.save_val:
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=d.keys())
                if not file_exists:
                    writer.writeheader()  # Write header only once if the file doesn't exist
                writer.writerow(d)

            os.makedirs(output_path + (f"/val_outputs/lcc/" if not args.awc_val else f"/awc_val_outputs/lcc/"), exist_ok=True)
            os.makedirs(output_path + (f"/val_outputs/preds/" if not args.awc_val else f"/awc_val_outputs/preds/"), exist_ok=True)
            _, gt_file_name, _ = get_folder_names(args.dataset)
            path2search = os.path.join(os.getenv('DATASET_PATH'), args.dataset, gt_file_name)
            gt_path = find_file_in_path(directory=path2search, filename=tst_filename)
            gt_f = nib.load(gt_path)

            save_val_2path(
                output_p=output_path + (f"/val_outputs/"),
                type_p="lcc",
                f_name=tst_filename,
                tensor=np.rot90(lcc, k=1, axes=(0, 1)),
                affine=gt_f.affine,
                header=gt_f.header
            )
            save_val_2path(
                output_p=output_path + (f"/val_outputs/"),
                type_p="preds",
                f_name=tst_filename,
                tensor=np.rot90(output_corr, k=1, axes=(0, 1)),
                affine=gt_f.affine,
                header=gt_f.header
            )
        self.test_step_outputs.append(d)
        return d

    def on_test_epoch_end(self):

        if args.training or dist.is_initialized():
            dist.barrier()
        # Aggregating metrics
        avg_metrics = {
            "avg_dice": np.mean([x["tst_dice"] for x in self.test_step_outputs]),
            "avg_iou": np.mean([x["tst_iou"] for x in self.test_step_outputs]),
            "avg_IoU": np.mean([x["tst_IoU"] for x in self.test_step_outputs]),
            "avg_DLR": np.mean([x["tst_DLR"] for x in self.test_step_outputs]),
            "avg_DBR": np.mean([x["tst_DBR"] for x in self.test_step_outputs]),
            "avg_Prec.": np.mean([x["tst_Prec."] for x in self.test_step_outputs]),
            "avg_Leak.": np.mean([x["tst_Leak."] for x in self.test_step_outputs]),
            "avg_AMR.": np.mean([x["tst_AMR."] for x in self.test_step_outputs]),
        }

        # Printing the average metrics
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")

        return avg_metrics


if __name__ == '__main__':

    if args.training:
        # initialize the lightning module
        net = SegmentationNet()
        # set up checkpoints
        model_path = os.path.join(output_path, "model")
        os.makedirs(model_path, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path,
            monitor="val_iou",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
            filename="best_metric_model_{epoch:02d}-{val_iou:.2f}",
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop_callback = EarlyStopping(monitor="val_iou", min_delta=0.002, patience=50, verbose=False, mode="max")

        print(f"The output path is {output_path}")
        print(f"The model_path is {model_path}")
        # initialize the trainer
        trainer = pl.Trainer(
            devices=-1,
            num_nodes=1,
            max_epochs=args.max_epochs,
            callbacks=[checkpoint_callback, lr_monitor],
            default_root_dir=model_path,
            log_every_n_steps=1,
            # profiler="simple",
            precision='16-mixed' if args.mixed else '32-true',
            enable_model_summary=True,
            max_time="00:23:00:00",
            deterministic=True,
            # accumulate_grad_batches=4,
            # reload_dataloaders_every_n_epochs=50,
            # fast_dev_run=1,
        )
        # train
        trainer.fit(net)
        print(f"train completed, best_metric: {net.best_val_dice:.4f} Dice and {net.best_val_iou:.4f} IoU " f"at epoch {net.best_val_epoch}")
        trainer.test(net, ckpt_path="best")

    if args.validation:
        # initialize the lightning module
        net = SegmentationNet()
        model_path = os.path.join(output_path, "model_AWC")
        os.makedirs(model_path, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path,
            monitor="val_iou",
            mode="max",
            save_top_k=3 if not args.awc else -1,
            save_last=True,
            auto_insert_metric_name=False,
            filename="best_metric_model_{epoch:02d}-{val_iou:.2f}" if not args.awc else "best_metric_awc_{epoch:02d}-{val_iou:.2f}",
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop_callback = EarlyStopping(monitor="val_iou", min_delta=0.002, patience=50, verbose=False, mode="max")

        # initialize the trainer
        trainer = pl.Trainer(
            devices=-1,
            num_nodes=1,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            max_epochs=0 if not args.awc else args.max_epochs,
            default_root_dir=model_path,
            log_every_n_steps=1,
            precision='16-mixed' if args.mixed else '32-true',
            enable_model_summary=True,
            num_sanity_val_steps=-1,
        )
        net.load_state_dict(torch.load(best_model_path)["state_dict"])
        # net.load_from_checkpoint(checkpoint_path=best_model_path)
        trainer.fit(net)
        trainer.test(net, ckpt_path=best_model_path if not args.awc else "best")
