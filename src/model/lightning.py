from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(pl.LightningModule):
    def __init__(self, 
                 net: nn.Module, 
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        
    def forward(self, x, p, rollout=1):
        t_eval = range(rollout+1)
        return self.net(x, p, t_eval=t_eval)
    
    def step(self, batch, batch_idx, stage):
        x, y, p = batch
        rollout = y.size(1)
        y_pred = self(x, p, rollout=rollout)
        loss = F.mse_loss(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.step(batch, batch_idx, "test")
        self.log("test_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer