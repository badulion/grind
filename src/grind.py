import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from findiff import FinDiff

import numpy as np
import matplotlib.pyplot as plt

from inverse_nufft import FourierInterpolator

from typing import List


class GrIND(nn.Module):
    def __init__(self, 
                 num_ks: int = 21,
                 grid_resolution: int = 64,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid_resolution = grid_resolution
        
        self.fourier_interpolator = FourierInterpolator(num_ks=num_ks, spatial_dim=2)
        self.interpolation_points, self.dx, self.dy = self.generate_interpolation_points(grid_resolution)
        
        self.solver_net = FinDiffNet()
        
        
    def generate_interpolation_points(self, grid_resolution):
        x_grid, y_grid = torch.meshgrid(torch.linspace(0, 1, grid_resolution), torch.linspace(0, 1, grid_resolution))
        p_grid = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)
        dx = x_grid[0, 1] - x_grid[0, 0]
        dy = y_grid[1, 0] - y_grid[0, 0]
        return p_grid, dx, dy
        
    def forward(self, x, p, t_eval=[1.0]):
        # interpolate on a grid
        x_grid = self.fourier_interpolator(p, x, self.interpolation_points)
        x_grid = x_grid.view(x.shape[0], self.grid_resolution, self.grid_resolution, x.shape[-1])
        x_grid = x_grid.permute(0, 3, 1, 2)
        
        # run solver
        x_pred = self.solver_net(x_grid, t_eval=t_eval)
        
        # interpolate back to the original points
        x_pred = x_pred.permute(0, 1, 3, 4, 2)
        x_pred = x_pred.reshape(len(t_eval), x.shape[0], self.grid_resolution**2, x.shape[-1])
        int_p_expanded = self.interpolation_points.view(1,1,*self.interpolation_points.shape).expand(len(t_eval), x.shape[0], -1,-1)
        x_pred = self.fourier_interpolator(self.interpolation_points.view(1,1,*self.interpolation_points.shape), x_pred, p)
        
        return x_pred
        
class FinDiffNet(nn.Module):
    def __init__(self, 
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        

    def forward(self, x, t_eval=[1.0]):
        return torch.stack([x]*len(t_eval), dim=0)
    
class GridStencil(nn.Module):
    def __init__(self, 
               input_dim: int,
               spatial_dim: int = 2,
               max_order: int = 2, 
               dx: List[float] = [1.0, 1.0],
               *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dx = dx
        self.spatial_dim = spatial_dim
        self.max_order = max_order
        self.input_dim = input_dim
                
        self.list_of_derivatives = GridStencil._generate_list_of_derivatives(num_dims=self.spatial_dim, max_order=self.max_order)
        self.list_of_derivatives = [self._generate_stencil(derivative) for derivative in self.list_of_derivatives]
    
    def _compositions(n, k):
        if n < 0 or k < 0: 
            return
        elif k == 0:
            if n == 0:
                yield []
            return
        elif k == 1:
            yield [n]
            return 
        else:
            for i in range(0,n+1):
                for comp in GridStencil._compositions(n-i, k-1):
                    yield [i] + comp

    def _generate_list_of_derivatives(num_dims = 2, max_order = 2):
        derivatives = []
        for i in range(1, max_order+1):
            derivatives += GridStencil._compositions(i, num_dims)
        return derivatives
    
    def _generate_stencil(self, derivative, accuracy=1):
        dx = 1
        dy = 1
        laplacian = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
        dif_op = FinDiff(*[(idx, dx, order) for idx, (dx, order) in enumerate(zip(self.dx, derivative)) if order > 0], acc=accuracy*2) 
        #dif_op = FinDiff((0,1,0) (1,1,1))
        
        #print(laplacian.stencil([50,50]))
        stencil = dif_op.stencil([accuracy*4+1]*2).data['C', 'C']
        #print(dir(stencil))
        print(stencil)
    
if __name__ == "__main__":
    m = GridStencil(2)
    print(m)