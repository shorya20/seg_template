#!/bin/bash
#SBATCH --gres=gpu:1                
#SBATCH --partition=gpgpuC         
#SBATCH --time=0-24:00:00           
#SBATCH --mem=24G                   
#SBATCH --cpus-per-task=6          
#SBATCH --mail-type=ALL             
#SBATCH --mail-user=ss2424          

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "------------------------------------------------------------"
source /vol/cuda/12.2.0/setup.sh
echo "CUDA version: $(nvcc --version | grep 'release' | awk '{print $6}')"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
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
export LUNGMASK_DEVICE=auto     # default
export LUNGMASK_BATCH=1         # safest on GPU
# --- USER CONFIGURATION -----
#    Example: "DiceLoss_ATM22_7_29_21_AttentionUNet_prtrFalse_AdamW_128_b1_p3_0.0001_alpha0.3_r_d1.0r_l0.7ReduceLROnPlateau"
EXP_PATH="DiceLoss_ATM22_8_31_20_AttentionUNet_prtrFalse_AdamW_128_b1_p10_0.0003_alpha0.3_r_d1.0r_l0.7WarmupCosineSchedule"

# 2. Set WEIGHTS_PATH to the name of the checkpoint file (.ckpt)
WEIGHTS_PATH="best_metric_model_48-0.4941.ckpt"
# --------------------------------------------------------------------
if [ -f config.env ]; then
  set -a
  source config.env
  set +a
fi

echo "Executing testing script with the following configuration:"
echo "  - Experiment Path: $EXP_PATH"
echo "  - Weights File:    $WEIGHTS_PATH"
echo "------------------------------------------------------------"

echo "Executing testing script to generate predictions for the official validation set..."
# python -m seg.training \
#     --dataset ATM22 \
#     --model_name BasicUNet \
#     --loss_func DiceCELoss\
#     --exp_path "$EXP_PATH" \
#     --weights_path "$WEIGHTS_PATH" \
#     --num_workers 0 \
#     --test_patch_size 128 \
#     --batch_size 1 \
#     --patch_size 128 \
#     --metric_testing \
#     --test_dataset val \

python -m seg.training \
    --dataset ATM22 \
    --model_name AttentionUNet \
    --loss_func DiceCELoss \
    --exp_path "$EXP_PATH" \
    --weights_path "$WEIGHTS_PATH" \
    --test_patch_size 128 \
    --batch_size 1 \
    --testing \
    --test_dataset val \
    --save_val \

# python data_prep/patch_audit.py \
#   --data_root ./ATM22/npy_files \
#   --output_dir ./ATM22/npy_files/patch_banks \
#   --patch_size 128 128 128 \
#   --stride 64 64 64 \
#   --max_patches_per_case 3000 \
#   --min_lung_coverage 0.05 \
#   --num_workers 6 \
    # python data_prep/patch_audit.py \
    #   --data_root ./ATM22/npy_files \
    #   --output_dir ./ATM22/npy_files/patch_banks_new \
    #   --patch_size 112 112 112 \
    #   --stride 56 56 56 \
    #   --min_lung_coverage 0.10 \
    #   --max_patches_per_case 3000 \
    #   --use_skan
# python plot_metrics.py outputs/DiceLoss_ATM22_.../model/lightning_logs/version_.../metrics_history.json
# python data_prep/convert_to_numpy.py \
#   --dataset_path ./ATM22 \
#   --batch1 TrainBatch1 \
#   --batch2 TrainBatch2_New \
#   --process_validation \
#   --generate_lungs \
#   --num_workers 8 \
#   --lungmask_workers 2 \
#   --lungmask_gpu_limit 1
    # python tools/check_intensity.py --root ./ATM22 --out ./ATM22/audit \
    #   --workers 8 --save_overlays --max_png 60
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Test run finished with exit code: $EXIT_CODE"
echo "Final memory usage: $(free -h)"
echo "Job finished at: $(date)"
echo "------------------------------------------------------------"

exit $EXIT_CODE
