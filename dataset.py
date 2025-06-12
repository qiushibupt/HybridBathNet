import logging
from typing import List, Tuple

import numpy as np
import rasterio
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    
    def __init__(self, file_paths: List[str], device: torch.device, normalization: bool = True):
        self.file_paths = file_paths
        self.device = device
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.file_paths)

    def interpolate_nan_band(self, band: np.ndarray) -> np.ndarray:
        x = np.arange(0, band.shape[1])
        y = np.arange(0, band.shape[0])
        xx, yy = np.meshgrid(x, y)
        
        valid_points = ~np.isnan(band)
        valid_xx = xx[valid_points]
        valid_yy = yy[valid_points]
        valid_values = band[valid_points]
        
        filled_band = griddata(
            (valid_xx, valid_yy), valid_values, (xx, yy), method='nearest'
        )
        
        return filled_band

    def fill_nans(self, image: np.ndarray) -> np.ndarray:
        filled_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            filled_image[i] = self.interpolate_nan_band(image[i])
        
        return filled_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            with rasterio.open(self.file_paths[idx]) as dataset:
                band_indexes = [2, 3, 4]
                image = dataset.read(out_dtype="float32", indexes=band_indexes)
                
                depth = dataset.read(out_dtype="float32")[-1]
                wdi = dataset.read(out_dtype="float32")[-2]
                
                depth_mask = ~np.isnan(depth)
                
                depth[np.isnan(depth)] = 0
                wdi[np.isnan(wdi)] = 0
                
                image_mask = ~np.isnan(image)
                image[np.isnan(image)] = 0
                
                mask = depth_mask.copy()
                for i in range(image_mask.shape[0]):
                    mask = mask & image_mask[i]
                
                mask = mask.astype(float)

                image = torch.tensor(image, device=self.device, dtype=torch.float32)
                depth = torch.tensor(depth, device=self.device, dtype=torch.float32)
                mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
                wdi = torch.tensor(wdi, device=self.device, dtype=torch.float32)
                
                return image, depth, mask, wdi
                
        except Exception as e:
            logging.error(f"Error loading sample {idx} from {self.file_paths[idx]}: {e}")
            raise
