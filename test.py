from src.model.grind import GrIND
from src.dataset.dynabench import DynabenchDataset
from torch.utils.data import DataLoader


ds = DynabenchDataset("train", "advection", "cloud", 4, "high", "data")
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

x, y, p = next(iter(dl))

m = GrIND(1, 2, 2, accuracy=1)

pred = m(x, p, t_eval=range(5)).detach().numpy()


# plot tricontour plots of the 4 rollout steps and the original x

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 5, figsize=(20, 5))

axs[0].tricontourf(p[0,:, 0], p[0,:, 1], x[0, :, 0], levels=100)
for i in range(4):
    axs[i+1].tricontourf(p[0,:, 0], p[0,:, 1], pred[0, i, :, 0], levels=100)    
plt.show()

