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
    datamodule.setup()

    dataloader = datamodule.test_dataloader()
    x, y, p = next(iter(dataloader))
    x = x.to("cuda")
    y = y.to("cuda")
    p = p.to("cuda")

    # #model = instantiate(cfg.model)
    from src.model.grind_neuralpde import GrIND_NeuralPDE
    ckpt_path = find_best_checkpoint("output/checkpoints/grind-neuralpde/wave-dt1.0-noise0.0-l20.0")
    model = GrIND_NeuralPDE.load_from_checkpoint(ckpt_path)
    
    # make prediction

    rollout = y.size(1)
    t_eval = range(rollout+1)

    preds = model(x, p, t_eval=t_eval)


    # create plots for each timestep in tmp folder
    import matplotlib.pyplot as plt
    import os
    import shutil
    import torch

    if os.path.exists("tmp"):
        shutil.rmtree("tmp")

    os.makedirs("tmp")

    p = p.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    
    images = []
    for i in range(rollout):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].tricontourf(p[0, :, 0], p[0, :, 1], preds[0, i, :, 0], levels=100)

        ax[1].tricontourf(p[0, :, 0], p[0, :, 1], y[0, i, :, 0], levels=100)

        plt.savefig(f"tmp/timestep_{i+1}.png")
        plt.close()
        # plt.figure(figsize=(10, 10))
        # plt.tricontourf(p[0, :, 0], p[0, :, 1], preds[0, i, :, 0], levels=100)
        # plt.savefig(f"tmp/pred/timestep_pred_{i+1}.png")
        # plt.close()

        # plt.figure(figsize=(10, 10))
        # plt.tricontourf(p[0, :, 0], p[0, :, 1], y[0, i, :, 0], levels=100)
        # plt.savefig(f"tmp/true/timestep_true_{i+1}.png")
        # plt.close()
        images.append(f"tmp/timestep_{i+1}.png")

        
    for i in range(rollout, rollout+8):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].tricontourf(p[0, :, 0], p[0, :, 1], preds[0, -1, :, 0], levels=100)

        ax[1].tricontourf(p[0, :, 0], p[0, :, 1], y[0, -1, :, 0], levels=100)

        plt.savefig(f"tmp/timestep_{i+1}.png")
        plt.close()
        # plt.figure(figsize=(10, 10))
        # plt.tricontourf(p[0, :, 0], p[0, :, 1], preds[0, i, :, 0], levels=100)
        # plt.savefig(f"tmp/pred/timestep_pred_{i+1}.png")
        # plt.close()

        # plt.figure(figsize=(10, 10))
        # plt.tricontourf(p[0, :, 0], p[0, :, 1], y[0, i, :, 0], levels=100)
        # plt.savefig(f"tmp/true/timestep_true_{i+1}.png")
        # plt.close()
        images.append(f"tmp/timestep_{i+1}.png")

    # create gif
    from PIL import Image

    fGIF = "animGIF.gif"
    frames = []

    for i in images:
        newImg = Image.open(i)
        frames.append(newImg)
    
    # Save into a GIF file that loops forever: duration is in milli-second
    frames[0].save(fGIF, format='GIF', append_images=frames[1:],
        save_all=True, duration=15, loop=0)


if __name__ == "__main__":
    main()