from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split

from .dynabench import DynabenchDataset
from dynabench.dataset.download import download_equation



class DynabenchDataModule(LightningDataModule):
    def __init__(self, 
                 base_path: str = "data",
                 equation: str = "advection",
                 structure: str = "cloud",
                 resolution: str = "low",
                 batch_size: int = 16, 
                 num_workers: int = 4,
                 train_rollout: int = 1,
                 val_rollout: int = 1,
                 test_rollout: int = 16):


        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.base_path = base_path
        self.equation = equation
        self.structure = structure
        self.resolution = resolution

        self.train_rollout = train_rollout
        self.val_rollout = val_rollout
        self.test_rollout = test_rollout




    def setup(self, stage=None):
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DynabenchDataset(
                split="train",
                equation=self.equation,
                structure=self.structure,
                resolution=self.resolution,
                base_path=self.base_path,
                rollout=self.train_rollout,
            )
            self.val_dataset = DynabenchDataset(
                split="val",
                equation=self.equation,
                structure=self.structure,
                resolution=self.resolution,
                base_path=self.base_path,
                rollout=self.val_rollout,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = DynabenchDataset(
                split="test",
                equation=self.equation,
                structure=self.structure,
                resolution=self.resolution,
                base_path=self.base_path,
                rollout=self.test_rollout,
            )

    def download_equation(self):
        download_equation(self.equation, self.structure, self.resolution, self.base_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)