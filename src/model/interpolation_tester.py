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
        
        self.fourier_interpolator = FourierInterpolator(num_ks=num_ks, spatial_dim=spatial_dim)
        
    def forward(self, x_cloud, p_cloud, p_grid):
        # check devices
        interpolation_points = p_grid.view(p_grid.shape[0], -1, 2)
            
        # interpolate on a grid
        x_grid = self.fourier_interpolator(p_cloud, x_cloud, interpolation_points)
        x_grid = x_grid.view(x_cloud.shape[0], p_grid.shape[1], p_grid.shape[2], x_cloud.shape[-1])
        x_grid = x_grid.permute(0, 3, 1, 2)
        return x_grid
        
        # resnet smoother
        if hasattr(self, "smoother"):
            x_grid = x_grid + self.smoother(x_grid)
        
        # run solver
        x_pred = self.solver_net(x_grid, t_eval=t_eval)
        
        # interpolate back to the original points
        x_pred = x_pred.permute(1, 0, 3, 4, 2)
        x_pred = x_pred.reshape(x.shape[0], len(t_eval[1:]), self.grid_resolution**2, x.shape[-1])
        x_pred = self.fourier_interpolator(self.interpolation_points.view(1,1,*self.interpolation_points.shape), x_pred, p.unsqueeze(1))
        return x_pred
        
    def step(self, batch, batch_idx, stage):
        (x_cloud, p_cloud, x_grid, p_grid) = batch
        x_interpolated = self(x_cloud, p_cloud, p_grid)
        loss = F.mse_loss(x_interpolated, x_grid)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "test")
        self.log("test_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

if __name__ == "__main__":
    pass