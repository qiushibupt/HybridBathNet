# HybridBathNet

HybridBathNet is a PyTorch implementation of a hybrid deep learning model for satellite-derived bathymetry estimation. The model combines satellite imagery with a pre-computed Water Depth Index (WDI) branch and uses multi-scale feature fusion to predict water depth.

## Overview

HybridBathNet is designed for shallow-water bathymetry prediction from multi-band remote sensing images.

The model includes:

- A U-Net-like encoder-decoder backbone for depth prediction
- A WDI prediction branch for auxiliary Water Depth Index supervision
- Multi-scale Feature Fusion Modules to combine image features and WDI-related features
- A masked loss function to ignore invalid or non-water pixels

The training script reports RMSE, R², and MAE on the test split, and saves the best model checkpoint according to R².

## Repository Structure

    HybridBathNet/
    ├── data/
    │   └── README.md          # Data organization instructions
    ├── dataset.py             # Dataset loader for multi-band GeoTIFF files
    ├── loss.py                # Masked depth and WDI loss
    ├── model.py               # HybridBathNet model definition
    ├── train.py               # Training and evaluation script
    ├── requirements.txt       # Python dependencies
    └── README.md

## Installation

Clone this repository:

    git clone https://github.com/qiushibupt/HybridBathNet.git
    cd HybridBathNet

Install the required dependencies:

    pip install -r requirements.txt

Main dependencies include:

- PyTorch
- NumPy
- Rasterio
- SciPy
- scikit-learn

## Data Preparation

The dataset should be organized under the data/ directory. Each study area or island should have its own folder.

Example:

    HybridBathNet/data/
    ├── Island_A/
    │   ├── 1.tif
    │   ├── 2.tif
    │   └── ...
    ├── Island_B/
    │   ├── 1.tif
    │   ├── 2.tif
    │   └── ...
    └── Island_C/
        ├── 1.tif
        ├── 2.tif
        └── ...

Each .tif file should contain multiple bands. The expected band order is:

    Satellite image bands
    ...
    Band n-1: WDI data
    Band n:   Ground-truth depth

Requirements:

- The last band should be the ground-truth water depth.
- The second-to-last band should be the pre-computed WDI.
- Invalid or non-water pixels should be set to NaN.
- Files in the same folder should have the same spatial size.
- float32 format is recommended.

For more details, see data/README.md.

## Training

Run training with:

    python train.py <dataset_name>

For example, if your data are stored in data/Island_A/, run:

    python train.py Island_A

Optional arguments:

    python train.py Island_A \
      --epochs 50 \
      --lr 0.0001 \
      --train-split 0.6 \
      --seed 42

Arguments:

| Argument | Description | Default |
|---|---|---|
| dataset | Dataset folder name under data/ | Required |
| --epochs | Number of training epochs | 50 |
| --lr | Learning rate | 0.0001 |
| --train-split | Fraction of data used for training | 0.6 |
| --seed | Random seed | 42 |

## Output

During training, the script prints evaluation metrics for each epoch:

- Loss
- RMSE
- R²
- MAE

Model checkpoints are saved to:

    HybridBathNet_pth/

Training logs are saved to:

    HybridBathNet.log

## Citation

If you use this code in your research, please cite our paper:

    @article{your2026hybridbathnet,
      title={HybridBathNet: ...},
      author={...},
      journal={...},
      year={2026}
    }
