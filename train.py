#!/usr/bin/env python3

import argparse
import csv
import datetime
import logging
import os
import random
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader

from model import HybridBathNet
from dataset import SatelliteDataset
from loss import masked_mse_loss


@dataclass
class TrainingConfig:
    n_channels: int = 3
    n_classes: int = 1
    num_epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 0.0001
    train_split: float = 0.6
    depth_loss_weight: float = 0.9
    wdi_loss_weight: float = 0.1
    seed: int = 42
    data_base_path: str = "./data"
    model_save_path: str = "./HybridBathNet_pth"
    log_file: str = "./HybridBathNet.log"

@dataclass
class TrainingResults:
    epoch: int
    loss: float
    rmse: float
    r2: float


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='a'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    logging.info(f"Random seed set to {seed}")


def load_and_split_data(data_path: str, train_split: float) -> Tuple[List[str], List[str]]:
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    all_files = [
        str(file_path) for file_path in data_dir.iterdir() 
        if file_path.is_file()
    ]
    
    if not all_files:
        raise ValueError(f"No files found in directory: {data_path}")
    
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    logging.info(f"Loaded {len(all_files)} files: {len(train_files)} train, {len(test_files)} test")
    
    return train_files, test_files


def evaluate_model(
    model: torch.nn.Module, 
    test_loader: DataLoader
) -> Tuple[float, float, float]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, depths, masks, wdi in test_loader:
            outputs, _ = model(images)
            
            outputs_np = outputs.squeeze().cpu().numpy()
            depths_np = depths.squeeze().cpu().numpy()
            mask_np = masks.squeeze().cpu().numpy().astype(bool)
            
            predictions.extend(outputs_np[mask_np])
            targets.extend(depths_np[mask_np])

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    return float(rmse), float(r2), float(mae)


def save_model(
    model_state_dict: Dict[str, Any], 
    save_path: str, 
    metrics: TrainingResults
) -> str:
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'HybridBathNet_{timestamp}_epoch{metrics.epoch}_r2{metrics.r2:.4f}.pth'
    filepath = save_dir / filename
    
    torch.save(model_state_dict, filepath)
    logging.info(f'Saved model to {filepath}')
    
    return str(filepath)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device
) -> Tuple[TrainingResults, TrainingResults, TrainingResults]:
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = TrainingResults(0, float('inf'), float('inf'), float('-inf'))
    best_rmse = TrainingResults(0, float('inf'), float('inf'), float('-inf'))
    best_r2 = TrainingResults(0, float('inf'), float('inf'), float('-inf'))
    
    best_model_state = None

    logging.info(f"Starting training for {config.num_epochs} epochs")
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (images, depths, masks, wdi) in enumerate(train_loader):
            optimizer.zero_grad()
            
            depth_pred, wdi_pred = model(images)
            
            loss = masked_mse_loss(
                depth_pred, depths, masks, wdi_pred, wdi,
                config.depth_loss_weight, config.wdi_loss_weight
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        rmse, r2, mae = evaluate_model(model, test_loader)
        
        print(f'Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.6f}, '
              f'RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
        
        current_result = TrainingResults(epoch + 1, float(avg_loss), rmse, r2)
        
        if avg_loss < best_loss.loss:
            best_loss = current_result
            
        if rmse < best_rmse.rmse:
            best_rmse = current_result
            
        if r2 > best_r2.r2:
            best_r2 = current_result
            best_model_state = model.state_dict().copy()

    logging.info(f'Training completed')
    logging.info(f'Best loss: Epoch {best_loss.epoch}, Loss: {best_loss.loss:.6f}, '
                f'RMSE: {best_loss.rmse:.4f}, R²: {best_loss.r2:.4f}')
    logging.info(f'Best RMSE: Epoch {best_rmse.epoch}, Loss: {best_rmse.loss:.6f}, '
                f'RMSE: {best_rmse.rmse:.4f}, R²: {best_rmse.r2:.4f}')
    logging.info(f'Best R²: Epoch {best_r2.epoch}, Loss: {best_r2.loss:.6f}, '
                f'RMSE: {best_r2.rmse:.4f}, R²: {best_r2.r2:.4f}')
    
    if best_model_state is not None:
        model_path = save_model(best_model_state, config.model_save_path, best_r2)
        logging.info(f'Best model saved with R²: {best_r2.r2:.4f} at epoch {best_r2.epoch}')

    return best_loss, best_rmse, best_r2


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train HybridBathNet for bathymetry prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', 
        help='Name of the dataset directory (relative to data_base_path)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0001, 
        help='Learning rate'
    )
    parser.add_argument(
        '--train-split', 
        type=float, 
        default=0.6, 
        help='Fraction of data to use for training'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_split=args.train_split,
        seed=args.seed
    )
    
    setup_logging(config.log_file)
    logging.info(f"Starting training script for dataset: {args.dataset}")
    logging.info(f"Configuration: {config}")
    
    set_random_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        data_path = os.path.join(config.data_base_path, args.dataset)
        train_files, test_files = load_and_split_data(data_path, config.train_split)
        
        train_dataset = SatelliteDataset(train_files, device, normalization=True)
        test_dataset = SatelliteDataset(test_files, device, normalization=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        model = HybridBathNet(
            n_channels=config.n_channels, 
            n_classes=config.n_classes
        ).to(device)
        
        logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        best_loss, best_rmse, best_r2 = train_model(
            model, train_loader, test_loader, config, device
        )
        
        logging.info(f"Training completed for dataset: {args.dataset}")
        logging.info(f"Final best results - Loss: {best_loss.loss:.6f}, "
                    f"RMSE: {best_rmse.rmse:.4f}, R²: {best_r2.r2:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    logging.info("Training script completed successfully")


if __name__ == "__main__":
    main()