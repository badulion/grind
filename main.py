import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import glob
import re
import numpy as np

from dynabench.dataset.download import download_equation

def find_best_checkpoint(ckpt_dir: str, metric: str = "val_loss", mode: str = "min"):
    files = glob.glob(ckpt_dir + f"/*{metric}*.ckpt")
    
    if len(files) == 0:
        return None
    
    metric_scores = [float(re.search(f"{metric}=(\d|\.)*\d", file)[0].split("=")[-1]) for file in files]
    
    if mode == "min":
        best_metric_idx = np.argmin(metric_scores)
    else:
        best_metric_idx = np.argmax(metric_scores)
    return files[best_metric_idx]
    


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    '''
    GrIND -> GRid Interpolation Network for Dynamical systems
    '''
    

    datamodule = instantiate(cfg.datamodule)

    model = instantiate(cfg.model)
    
    trainer = instantiate(cfg.trainer)
    
    checkpoint_path = find_best_checkpoint(cfg.trainer.callbacks[0].dirpath)
    #trainer.fit(model, datamodule, ckpt_path=cfg.train_from_checkpoint)
    trainer.test(model, datamodule, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    main()