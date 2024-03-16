import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

from typing import List

from .modules import CNNSolver
from .cnn import SimpleCNN
from .inverse_nufft import FourierInterpolator


class GrIND_NeuralPDE(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 cnn_hidden_size: int = 64,
                 cnn_hidden_layers: int = 1,
                 smoother_hidden_size: int = 64,
                 smoother_hidden_layers: int = 4,
                 smoother_kernel_size: int = 7,
                 num_ks: int = 21,
                 grid_resolution: int = 64,
                 use_adjoint: bool = False,
                 use_smoother: bool = False,
                 use_instance_norm: bool = True,
                 solver: dict = {"method": "dopri5"},
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 training_noise: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.grid_resolution = grid_resolution
        
        self.fourier_interpolator = FourierInterpolator(num_ks=num_ks, spatial_dim=2)
        self.interpolation_points = self.generate_interpolation_points(grid_resolution)
        if use_smoother:
            self.smoother = SimpleCNN(input_size=input_dim, 
                                    hidden_layers=smoother_hidden_size,
                                    hidden_channels=smoother_hidden_layers,
                                    kernel_size=smoother_kernel_size)
        
        self.solver_net = CNNSolver(
            input_dim=input_dim,
            cnn_hidden_layers=cnn_hidden_layers,
            cnn_hidden_size=cnn_hidden_size,
            solver=solver,
            use_adjoint=use_adjoint
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
        x, y, p = batch
        rollout = y.size(1)
        t_eval = range(rollout+1)
        if stage == "train":
            noise_strength = np.random.uniform(0, self.hparams.training_noise) 
            x += torch.randn_like(x) * noise_strength
        y_pred = self(x, p, t_eval=t_eval)
        loss = F.mse_loss(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "test")
        stepwise_mse = (y_pred - y).pow(2).mean(dim=[0,2,3,4])
        for i, mse in enumerate(stepwise_mse):
            self.log(f"test_loss_t{i+1}", mse)
        self.log("test_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__":
    pass