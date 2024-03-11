from dynabench.dataset import DynabenchIterator

import torch
import numpy as np

class DynabenchInterpolator(torch.utils.data.Dataset):
    def __init__( 
            self,
            split: str="train",
            equation: str="wave",
            resolution: str="high",
            base_path: str="data",
            **kwargs):
        
        self.split = split
        self.equation = equation
        self.resolution = resolution
        self.base_path = base_path

        self.iterator_cloud = DynabenchIterator(
            base_path=self.base_path,
            split=self.split,
            equation=self.equation,
            structure="cloud",
            resolution=self.resolution,
            rollout=1,
        )

        self.iterator_full = DynabenchIterator(
            base_path=self.base_path,
            split=self.split,
            equation=self.equation,
            structure="grid",
            resolution="full",
            rollout=1,
        )

    
    def __len__(self):
        return len(self.iterator_cloud)
    
    def __getitem__(self, index):

        x_cloud, _, p_cloud = self.iterator_cloud[index]
        x_grid, _, p_grid = self.iterator_full[index]


        return x_cloud[-1].astype(np.float32), p_cloud.astype(np.float32), \
               x_grid[-1].astype(np.float32), p_grid.astype(np.float32)
               
               
