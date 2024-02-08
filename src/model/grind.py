import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np


from typing import List

from .modules import FinDiffNet
from .inverse_nufft import FourierInterpolator


class GrIND(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 spatial_dim: int = 2,
                 max_order: int = 2,
                 accuracy: int = 1,
                 symnet_hidden_size: int = 64,
                 symnet_hidden_layers: int = 1,
                 num_ks: int = 21,
                 grid_resolution: int = 64,
                 solver: dict = {"method": "dopri5"},
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.grid_resolution = grid_resolution
        
        self.fourier_interpolator = FourierInterpolator(num_ks=num_ks, spatial_dim=2)
        self.interpolation_points = self.generate_interpolation_points(grid_resolution)
        
        self.solver_net = FinDiffNet(
            input_dim=input_dim,
            spatial_dim=spatial_dim,
            max_order=max_order,
            dx=[1/grid_resolution]*spatial_dim,
            accuracy=accuracy,
            symnet_hidden_size=symnet_hidden_size,
            symnet_hidden_layers=symnet_hidden_layers,
            solver=solver
        )
        
        
    def generate_interpolation_points(self, grid_resolution):
        x_grid, y_grid = torch.meshgrid(torch.linspace(0, 1, grid_resolution), torch.linspace(0, 1, grid_resolution))
        p_grid = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)
        return p_grid
        
    def forward(self, x, p, t_eval=[0.0, 1.0]):
        # check devices
        if self.interpolation_points.device != p.device:
            self.interpolation_points = self.interpolation_points.to(p.device)
            
        # interpolate on a grid
        x_grid = self.fourier_interpolator(p, x, self.interpolation_points)
        x_grid = x_grid.view(x.shape[0], self.grid_resolution, self.grid_resolution, x.shape[-1])
        x_grid = x_grid.permute(0, 3, 1, 2)
        
        # run solver
        x_pred = self.solver_net(x_grid, t_eval=t_eval)
        
        # interpolate back to the original points
        x_pred = x_pred.permute(1, 0, 3, 4, 2)
        x_pred = x_pred.reshape(x.shape[0], len(t_eval[1:]), self.grid_resolution**2, x.shape[-1])
        int_p_expanded = self.interpolation_points.view(1,1,*self.interpolation_points.shape).expand(len(t_eval), x.shape[0], -1,-1)
        x_pred = self.fourier_interpolator(self.interpolation_points.view(1,1,*self.interpolation_points.shape), x_pred, p.unsqueeze(1))
        return x_pred
        
    def step(self, batch, batch_idx, stage):
        x, y, p = batch
        rollout = y.size(1)
        t_eval = range(rollout+1)
        y_pred = self(x, p, t_eval=t_eval)
        loss = F.mse_loss(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "test")
        self.log("test_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer


    

if __name__ == "__main__":
    pass