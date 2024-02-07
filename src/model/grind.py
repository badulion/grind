import torch

import numpy as np
import matplotlib.pyplot as plt


from typing import List

from .modules import FinDiffNet, GridStencil
from .inverse_nufft import FourierInterpolator

class GrIND(torch.nn.Module):
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
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        



    

if __name__ == "__main__":
    
    from dynabench.dataset.iterator import DynabenchIterator
    
    ds = DynabenchIterator("train", equation="kuramotosivashinsky", structure="grid", resolution="full")
    
    x = np.concatenate([ds[18][0], ds[1234][0], ds[156][0]], axis=1)
    x = torch.tensor(x, dtype=torch.float32)
    
    m = FinDiffNet(3, spatial_dim=2, max_order=2, dx=[1.0, 1.0], accuracy=1)
    ders = m(x, t_eval=[0, 4e4]).detach().numpy()[-1]
    print(ders.shape)

    
    # print(x.shape)
    # m = GridStencil(3, accuracy=1)
    # ders = m(x)
    
    import matplotlib.pyplot as plt
    
    # plot the original data in first row and the dx derivative in the second row
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        ax[0, i].imshow(x[0, i, :, :])
        ax[1, i].imshow(ders[0,i, :, :])
    plt.show()