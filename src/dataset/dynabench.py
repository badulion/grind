from dynabench.dataset import DynabenchIterator

import torch
import numpy as np

class DynabenchDataset(torch.utils.data.Dataset):
    def __init__( 
            self,
            split: str="train",
            equation: str="wave",
            structure: str="cloud",
            rollout: int = 16,
            resolution: str="low",
            base_path: str="data",
            **kwargs):
        
        self.split = split
        self.equation = equation
        self.structure = structure
        self.resolution = resolution
        self.rollout = rollout
        self.base_path = base_path

        self.iterator = DynabenchIterator(
            base_path=self.base_path,
            split=self.split,
            equation=self.equation,
            structure=self.structure,
            resolution=self.resolution,
            rollout=self.rollout,
            lookback=16
        )

    
    def __len__(self):
        return len(self.iterator)
    
    def __getitem__(self, index):

        x, y, p = self.iterator[index]


        return x[-1].astype(np.float32), y.astype(np.float32), p.astype(np.float32)