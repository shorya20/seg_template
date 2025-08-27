#!/bin/bash
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --partition=gpgpuC          # GPU partition
#SBATCH --time=0-24:00:00           # Reduced time limit (4 hours)
#SBATCH --mem=24G                   # Keep the same memory allocation
#SBATCH --cpus-per-task=6          # Same CPU allocation
#SBATCH --mail-type=ALL             # Email notifications
#SBATCH --mail-user=ss2424          # Your username/email

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "------------------------------------------------------------"
source /vol/cuda/12.2.0/setup.sh
echo "CUDA version: $(nvcc --version | grep 'release' | awk '{print $6}')"
PROJECT_DIR="/vol/bitbucket/ss2424/project/seg_template"
cd $PROJECT_DIR
echo "Changed directory to: $(pwd)"
echo "Activating venv: source project/bin/activate"
source project/bin/activate
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# PyTorch 2.6.0 uses bundled CUDA libraries - add minimal support
echo "Loading minimal CUDA runtime for PyTorch"
echo "CUDA LOADED"

# Check GPU allocation
echo "Running nvidia-smi:"
/usr/bin/nvidia-smi
echo "------------------------------------------------------------"

# --- USER CONFIGURATION: Set your experiment and weights path here ---
# 1. Set EXP_PATH to the name of your experiment folder inside 'outputs/'.
#    Example: "DiceLoss_ATM22_7_29_21_AttentionUNet_prtrFalse_AdamW_128_b1_p3_0.0001_alpha0.3_r_d1.0r_l0.7ReduceLROnPlateau"
EXP_PATH="DiceLoss_ATM22_8_15_22_AttentionUNet_prtrFalse_AdamW_128_b1_p7_3e-05_alpha0.3_r_d1.0r_l0.7CosineAnnealingWarmRestarts"

# 2. Set WEIGHTS_PATH to the name of the checkpoint file (.ckpt) you want to use for inference.
WEIGHTS_PATH="best_metric_model_31-0.3396.ckpt"
# --------------------------------------------------------------------

echo "Executing testing script with the following configuration:"
echo "  - Experiment Path: $EXP_PATH"
echo "  - Weights File:    $WEIGHTS_PATH"
echo "------------------------------------------------------------"

echo "Executing testing script to generate predictions for the official validation set..."
# python -m seg.training \
#     --dataset ATM22 \
#     --model_name AttentionUNet \
#     --loss_func DiceLoss \
#     --exp_path "$EXP_PATH" \
#     --weights_path "$WEIGHTS_PATH" \
#     --test_patch_size 128 \
#     --batch_size 1 \
#     --metric_testing \
#     --test_dataset val \
#     --save_val

# python -m seg.training \
#     --dataset ATM22 \
#     --model_name AttentionUNet \
#     --loss_func DiceLoss \
#     --exp_path "$EXP_PATH" \
#     --weights_path "$WEIGHTS_PATH" \
#     --test_patch_size 128 \
#     --batch_size 1 \
#     --metric_testing \
#     --test_dataset val \
#     --save_val
python data_prep/patch_audit.py \
  --data_root ./ATM22/npy_files \
  --output_dir ./ATM22/npy_files/patch_banks \
  --patch_size 128 128 128 \
  --stride 64 64 64 \
  --min_lung_coverage 0.10 \
  --max_patches_per_case 3000 \
  --use_skan
# python plot_metrics.py outputs/DiceLoss_ATM22_.../model/lightning_logs/version_.../metrics_history.json
# python data_prep/convert_to_numpy.py --dataset_path ./ATM22 --batch1 TrainBatch1 --batch2 TrainBatch2 --process_validation --generate_lungs --num_workers 6
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Test run finished with exit code: $EXIT_CODE"
echo "Final memory usage: $(free -h)"
echo "Job finished at: $(date)"
echo "------------------------------------------------------------"

# Optional: Run alignment verification on a sample prediction
# Uncomment and adjust paths as needed
# echo "Verifying alignment of sample prediction..."
# python -m seg.verify_alignment \
#     --original /path/to/original/ATM_XXX_0000.nii.gz \
#     --prediction output/preds/ATM_XXX/ATM_XXX_seg.nii.gz \
#     --metadata ATM22/npy_files/imagesTr/ATM_XXX_0000_meta.json

exit $EXIT_CODE