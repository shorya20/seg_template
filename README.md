# Create a README.md file for the user's project with dataset links and step-by-step instructions.
readme = r"""# Airway Segmentation (ATM22) — Non‑Uniform Patch Sampling Project

This repository contains a 3D airway segmentation pipeline for the **ATM22 (Airway Tree Modeling 2022)** datasets with an emphasis on **non‑uniform patch sampling** and **topology‑aware training** (e.g., MONAI 1.5.0 `SoftclDiceLoss`). It includes SLURM-ready scripts for HPC, as well as Python-only commands for local runs.

---

## 1) Download the ATM22 datasets

Please download the three official ATM22 splits below and extract them under a single data root, e.g. `data/ATM22/`:

- **Train Batch 1:** https://zenodo.org/record/6590745  
- **Train Batch 2:** https://zenodo.org/record/6590774  
- **Validation Set:** https://zenodo.org/record/6590026

A typical layout after extraction (paths are examples; actual file names may differ):
data/
└── ATM22/
    ├── TrainBatch1/
    │   ├── imagesTr/         
    │   └── labelsTr/           
    ├── TrainBatch2/
    │   ├── imagesTr/
    │   └── labelsTr/
    └── Validation/
        ├── imagesVal/

## 2) Environment setup

### Option A — Python venv (recommended; matches your request)

```bash
# from the repository root
python3 -m venv project
source project/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# OR CPU-only (no GPU)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
