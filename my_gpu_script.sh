#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuC
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ss2424

# Print job info
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep 'Mem:' | awk '{print $2}')"
echo "------------------------------------------------------------"

# Configure the environment
cd /vol/bitbucket/ss2424/project/seg_template
echo "Changed directory to: $(pwd)"

echo "Attempting to activate venv: source project/bin/activate"
source project/bin/activate

# Print Python environment info
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Load CUDA
source /vol/cuda/12.2.0/setup.sh
echo "CUDA version: $(nvcc --version | grep 'release' | awk '{print $6}')"

# Check GPU allocation
echo "Running nvidia-smi:"
/usr/bin/nvidia-smi
echo "------------------------------------------------------------"

# Clean up Persistent cache 
PERSISTENT_CACHE_DIR_PATH="./ATM22/npy_files/persistent_cache" # Define it once
SHOULD_CLEAN_CACHE=${CLEAN_CACHE:-false} # Default to false, set CLEAN_CACHE=true env var to clear

echo "Persistent cache directory: $PERSISTENT_CACHE_DIR_PATH"
if [ "$SHOULD_CLEAN_CACHE" = true ]; then
    echo "CLEAN_CACHE is true. Ensuring Persistent cache directory exists and is clean."
    if [ -d "$PERSISTENT_CACHE_DIR_PATH" ]; then
        echo "Removing contents of $PERSISTENT_CACHE_DIR_PATH..."
        rm -rf "$PERSISTENT_CACHE_DIR_PATH"/*  # Remove contents, not the directory itself
        echo "Cache directory cleaned."
    else
        echo "Cache directory $PERSISTENT_CACHE_DIR_PATH does not exist. Creating it."
        mkdir -p "$PERSISTENT_CACHE_DIR_PATH"
    fi
else
    echo "CLEAN_CACHE is false or not set. Persistent cache will NOT be cleared. Creating directory if it doesn't exist."
    mkdir -p "$PERSISTENT_CACHE_DIR_PATH" # Ensure it exists
fi


SKEL_DIR_PATH="./ATM22/npy_files/skeletons"
REGIONS_DIR_PATH="./ATM22/npy_files/regions"
echo "Ensuring skeleton directory exists: $SKEL_DIR_PATH"
mkdir -p "$SKEL_DIR_PATH"
echo "Ensuring regions directory exists: $REGIONS_DIR_PATH"
mkdir -p "$REGIONS_DIR_PATH"

echo "------------------------------------------------------------"
echo "Memory usage before training: $(free -h)"
echo "Executing Python script: python -m seg.training"
python -m seg.training \
    --dataset ATM22 \
    --model_name AttentionUNet \
    --patch_size 128 \
    --loss_func DiceLoss \
    --resume_from_checkpoint outputs/DiceLoss_ATM22_8_9_11_AttentionUNet_prtrFalse_AdamW_128_b1_p7_0.0001_alpha0.3_r_d1.0r_l0.7ReduceLROnPlateau/model/best_metric_model_22-0.7134.ckpt \
    --optimizer AdamW \
    --num_workers 1 \
    --mixed \
    --training \
    --max_epochs 50

# Option 2: No cache (uncomment this and comment above if still OOM)
# python -m seg.training \
#     --dataset ATM22 \
#     --model_name AttentionUNet \
#     --optimizer AdamW \
#     --lr 0.0003 \
#     --mixed \
#     --training \
#     --max_epochs 100 \
#     --loss_func TverskyLoss \
#     --alpha 0.2 \
#     --batch_size 2\
#     --out_channels 1 \
#     --dropout 0.2 \
#     --patch_size 96 \
#     --num_workers 0 \
#     --prefetch_factor 1

# --gradient_checkpointing \
# Capture the exit code of your Python script
EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "Python script finished with exit code: $EXIT_CODE"
echo "Final memory usage: $(free -h)"
echo "Job finished at: $(date)"
echo "------------------------------------------------------------"

# Exit with the Python script's exit code - important for Slurm status
exit $EXIT_CODE
