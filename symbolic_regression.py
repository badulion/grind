import juliacall
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from pysr import PySRRegressor
from src.model.grind import GrIND

from src.dataset.dynabench import DynabenchDataset

from typing import List, Tuple

class ListModule(torch.nn.Module):
    def __init__(self, list_of_modules: List[torch.nn.Module], *args):
        super(ListModule, self).__init__()
        self.list_of_modules = list_of_modules
        
    def forward(self, x):
        preds = []
        x_flat = x.reshape(-1, x.shape[-1])
        for module in self.list_of_modules:
            preds.append(module(x_flat))
        preds = torch.cat(preds, dim=-1)
        return preds.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        
        
def generate_variable_names(variables: List[str], derivative_orders: List[List[int]]) -> List[str]:
    suffixes = [f"_d_dx{t[0]}_dy{t[1]}" for t in derivative_orders]
    return variables + [f"{var}{suffix}" for var in variables for suffix in suffixes]

var_names = generate_variable_names(["p", "T", "v", "w"], [[0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])


eq = "gasdynamics"

ds = DynabenchDataset(equation=eq, structure="cloud", resolution="high")
dl = DataLoader(ds, batch_size=4, shuffle=True)
x, y, p = next(iter(dl))

ckpt_path = "output/checkpoints/grind/gasdynamics-cloud-high-order2/epoch=34-step=35000-val_loss=0.30.ckpt"

#ckpt_path = "output/checkpoints/grind/advection-cloud-high-order2/epoch=10-step=11000-val_loss=0.04.ckpt"

model = GrIND.load_from_checkpoint(ckpt_path)


fouriernet = model.fourier_interpolator
findiff = model.solver_net.findiff
x_grid = fouriernet(p, x, model.interpolation_points)
x_grid = x_grid.view(x.shape[0], model.grid_resolution, model.grid_resolution, x.shape[-1])
x_grid = x_grid.permute(0, 3, 1, 2)

x_derivatives = findiff(x_grid)
x_features = torch.cat([x_grid, x_derivatives], dim=1)
if hasattr(model.solver_net, "instance_norm"):
    x_features = model.solver_net.instance_norm(x_features)
features = x_features.permute(0,2,3,1)
targets = model.solver_net.symnet(features)

features = features.reshape(-1, features.shape[-1]).detach().numpy()
targets = targets.reshape(-1, targets.shape[-1]).detach().numpy()


if not os.path.exists("output/sr"):
    os.makedirs("output/sr")

sr_model = PySRRegressor(
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "*", "/", "-"],
    maxsize=40,
    unary_operators=[
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    equation_file=f"output/sr/{eq}.csv",
    model_selection='score',
    verbosity=1
)

random_indices = np.random.choice(range(len(features)), 1000, replace=False)

sr_model.fit(features[random_indices,:], targets[random_indices, 0],variable_names=var_names,)

print(sr_model)

# torch_model = ListModule(sr_model.pytorch())

# model.solver_net.symnet = torch_model


# import pytorch_lightning as pl
# trainer = pl.Trainer(limit_test_batches=10)
# trainer.test(model, dl)

