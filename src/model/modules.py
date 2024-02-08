import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
from findiff import FinDiff

from typing import List, Optional

class FinDiffNet(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 spatial_dim: int = 2,
                 max_order: int = 2,
                 dx: List[float] = [1.0, 1.0],
                 accuracy: int = 1,
                 symnet_hidden_size: int = 64,
                 symnet_hidden_layers: int = 1,
                 solver: dict = {"method": "dopri5"},
                 use_adjoint: bool = False,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.findiff = GridStencil(input_dim, 
                                   spatial_dim=spatial_dim, 
                                   max_order=max_order, 
                                   dx=dx, 
                                   accuracy=accuracy)
        self.num_derivatives = self.findiff.num_derivatives
        self.instance_norm = nn.InstanceNorm2d(input_dim*self.num_derivatives, momentum = 0.01)
        self.symnet = MLP(input_size=input_dim*self.num_derivatives, 
                          output_size=input_dim, 
                          hidden_size=symnet_hidden_size, 
                          hidden_layers=symnet_hidden_layers)

        self.solver = solver
        self.use_adjoint = use_adjoint
    def _ode(self, t, x):
        x = self.findiff(x)
        x = self.instance_norm(x)
        x = x.permute(0,2,3,1)
        x = self.symnet(x)
        x = x.permute(0,3,1,2)
        return x
    
    def forward(self, x, t_eval=[0.0, 1.0]):
        t_eval = torch.tensor(t_eval, dtype=x.dtype, device=x.device)
        if self.use_adjoint:
            return odeint_adjoint(self._ode, x, t_eval, **self.solver, adjoint_params=self.symnet.parameters())[1:]
        else:
            return odeint(self._ode, x, t_eval, **self.solver)[1:]

class MLP(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_size: int = 64, 
                 hidden_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers-1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        # initialize layers
        nn.init.normal_(self.input_layer.weight, std = 0.001)
        nn.init.zeros_(self.input_layer.bias)
        
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, std = 0.001)
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.output_layer.weight, std = 0.001)
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x

class ConvMLPSymNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers=1, *args, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.input_layer = torch.nn.Conv2d(input_size, hidden_size, kernel_size=1)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=1) for _ in range(hidden_layers-1)])
        self.output_layer = torch.nn.Conv2d(hidden_size, output_size, kernel_size=1)
        self.activation = torch.nn.ReLU()

        # initialize layers
        torch.nn.init.normal_(self.input_layer.weight, std = 0.001)
        torch.nn.init.zeros_(self.input_layer.bias)
        
        for layer in self.hidden_layers:
            torch.nn.init.normal_(layer.weight, std = 0.001)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.normal_(self.output_layer.weight, std = 0.001)
        torch.nn.init.zeros_(self.output_layer.bias)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x
    
class GridStencil(nn.Module):
    def __init__(self, 
               input_dim: int,
               spatial_dim: int = 2,
               max_order: int = 2, 
               dx: List[float] = [1.0, 1.0],
               accuracy: int = 1,
               *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dx = dx
        self.spatial_dim = spatial_dim
        self.max_order = max_order
        self.input_dim = input_dim
        self.accuracy = accuracy
                
        self.list_of_derivative_orders = GridStencil._generate_list_of_derivatives(num_dims=self.spatial_dim, max_order=self.max_order)
        self.num_derivatives = len(self.list_of_derivative_orders)
        self.list_of_stencils = [self._generate_stencil(derivative, accuracy=accuracy) for derivative in self.list_of_derivative_orders]
        
        # calculate filter kernel size
        self.filter_size = max([max([max([abs(s) for s in positions]) for positions in stencil.keys()]) for stencil in self.list_of_stencils])
    
        self.list_of_filters = [self._generate_stencil_filter(stencil) for stencil in self.list_of_stencils]
        self.single_findiff_filter = torch.stack(self.list_of_filters, dim=0)
        self.filter = self._generate_convolution(self.single_findiff_filter)
        
    def forward(self, x):
        # check devices
        if self.filter.device != x.device:
            self.filter = self.filter.to(x.device)
        
        x_padded = F.pad(x, [self.filter_size]*self.spatial_dim*2, mode="circular")
        return F.conv2d(x_padded, self.filter)
    
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
        dif_op = FinDiff(*[(idx, dx, order) for idx, (dx, order) in enumerate(zip(self.dx, derivative)) if order > 0], acc=accuracy*2) 
        #dif_op = FinDiff((0,1,0) (1,1,1))
        
        #print(laplacian.stencil([50,50]))
        stencil = dif_op.stencil([(accuracy+self.max_order)**2]*2).data['C', 'C']
        #print(dir(stencil))
        return stencil
    
    def _generate_stencil_filter(self, stencil):
        findiff_filter = torch.zeros([self.filter_size*2+1]*self.spatial_dim)
        for position, value in stencil.items():
            position = tuple(p + self.filter_size for p in position)
            findiff_filter[position] = value
        return findiff_filter
    
    def _generate_convolution(self, filter):
        zero_filter = torch.zeros_like(filter)
        per_variable_convolutions = []
        for i in range(self.input_dim):
            single_variable_convolution = [zero_filter]*self.input_dim
            single_variable_convolution[i] = filter
            single_variable_convolution = torch.cat(single_variable_convolution, dim=0)
            per_variable_convolutions.append(single_variable_convolution)
        return torch.stack(per_variable_convolutions, dim=1)
    
    
    
if __name__ == "__main__":
    # test the findiffnet layer on a random tensor
    x = torch.randn(1,3,64,64)
    m = FinDiffNet(3, 2)
    
    pred = m(x)