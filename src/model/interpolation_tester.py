import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

from typing import List

from .modules import CNNSolver
from .cnn import SimpleCNN
from .inverse_nufft import FourierInterpolator


class InterpolationTester(pl.LightningModule):
    def __init__(self, 
                 spatial_dim: int = 2,
                 num_ks: int = 21,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.min_ks = 3
        self.max_ks = 25
        
    def forward(self, x_cloud, p_cloud, p_grid, ks: int = 10):
        # check devices
        interpolation_points = p_grid.view(p_grid.shape[0], -1, 2)

        # interpolation layer 
        interpolation_layer = FourierInterpolator(num_ks=ks, spatial_dim=2)
            
        # interpolate on a grid
        x_grid = interpolation_layer(p_cloud, x_cloud, interpolation_points)
        x_grid = x_grid.view(x_cloud.shape[0], p_grid.shape[1], p_grid.shape[2], x_cloud.shape[-1])
        x_grid = x_grid.permute(0, 3, 1, 2)
        return x_grid
        
    def step(self, batch, batch_idx, stage):
        (x_cloud, p_cloud, x_grid, p_grid) = batch
        losses = {}
        for ks in range(self.min_ks, self.max_ks+1):
            x_interpolated = self(x_cloud, p_cloud, p_grid, ks=ks)
            loss = F.mse_loss(x_interpolated, x_grid)
            losses[f"ks_{ks}"] = loss
        return losses
    
    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx, "train")

        for key, value in losses.items():
            self.log(f"train_{key}", value)
        return losses["ks_10"]
    
    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx, "val")
        for key, value in losses.items():
            self.log(f"val_{key}", value)
        return losses["ks_10"]
    
    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx, "test")
        for key, value in losses.items():
            self.log(f"test_{key}", value)
        return losses["ks_10"]
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

if __name__ == "__main__":
    pass