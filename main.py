import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    '''
    GrIND -> GRid Interpolation Network for Dynamical systems
    '''
    ds = instantiate(cfg.dataset)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
    x, y, p = next(iter(dl))
    
    from src.inverse_nufft import FourierInterpolator
    
    fi = FourierInterpolator(p, x, num_ks=25)
    x_hat = fi(p)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    im1 = ax[0].tricontourf(p[-1, :,0], p[-1,:,1], x[-1,:,0], levels=100)
    fig.colorbar(im1)
    im2 = ax[1].tricontourf(p[-1, :,0], p[-1,:,1], x_hat[-1,:,0], levels=100)
    fig.colorbar(im2)
    im3 = ax[2].tricontourf(p[-1, :,0], p[-1,:,1], (x[-1,:,0]-x_hat[-1,:,0])**2, levels=100)
    fig.colorbar(im3)
    plt.show()

if __name__ == "__main__":
    main()