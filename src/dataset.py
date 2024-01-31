from dynabench.dataset.iterator import DynabenchIterator
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl


class Dynabench(Dataset):
    def __init__(
        self,
        split: str,
        equation: str,
        structure: str,
        resolution: str,
    ):
        super().__init__()

        self.iterator = DynabenchIterator(
            split=split,
            equation=equation,
            structure=structure,
            resolution=resolution,
        )
        
    def __len__(self):
        return len(self.iterator)
    
    def __getitem__(self, idx):
        x, y, points = self.iterator[idx]
        return x[0].astype(np.float32), y.astype(np.float32), points.astype(np.float32)
    
    
class DynabenchDatamodule(pl.LightningDataModule):
    def __init__(self, equation: str = "wave", 
                       structure: str = "cloud", 
                       resolution: str = "high", 
                       batch_size: int = 32, 
                       num_workers: int = 8,
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.equation = equation
        self.structure = structure
        self.resolution = resolution
        self.num_workers = num_workers
        
        
    def setup(self, stage=None):
        self.train_ds = Dynabench("train", equation=self.equation, structure=self.structure, resolution=self.resolution)
        self.val_ds = Dynabench("val", equation=self.equation, structure=self.structure, resolution=self.resolution)
        self.test_ds = Dynabench("test", equation=self.equation, structure=self.structure, resolution=self.resolution)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    