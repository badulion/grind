import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

def reconstruction_mse(x, p, x_full, p_full, num_ks=21):
    '''
    Reconstruction MSE of the GrIND model
    '''
    from src.inverse_nufft import FourierInterpolator
    import torch
    fi = FourierInterpolator(p, x, num_ks=num_ks)
    p_full = p_full.reshape(-1, 2)
    x_hat = fi(p_full)
    x_hat = x_hat.reshape(x_full.shape)
    return torch.nn.functional.mse_loss(x_hat, x_full).item()

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    '''
    GrIND -> GRid Interpolation Network for Dynamical systems
    '''
    ds = instantiate(cfg.dataset)
    ds_full = instantiate(cfg.dataset, resolution="full", structure="grid")

    from torch.utils.data import DataLoader
    import torch
    x, y, p = ds[16]
    x = torch.tensor(x)
    y = torch.tensor(y)
    p = torch.tensor(p)
    
    x_full, y_full, p_full = ds_full[16]
    x_full = torch.tensor(x_full)
    y_full = torch.tensor(y_full)
    p_full = torch.tensor(p_full)
    
    # mse = reconstruction_mse(x, p, x_full, p_full, num_ks=21)
    # ks_list = range(1, 50)
    # mse_list = [reconstruction_mse(x, p, x_full, p_full, num_ks=ks) for ks in ks_list]
    # import matplotlib.pyplot as plt
    # plt.plot(ks_list, mse_list)
    # plt.ylim(0, 0.5)
    # plt.xlabel("Number of Fourier modes")
    # plt.ylabel("Reconstruction MSE")
    # plt.show()
    
    from src.inverse_nufft import FourierInterpolator
    
    fi = FourierInterpolator(p, x, num_ks=20)
    # interpolate on a grid
    x_grid, y_grid = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64))
    p_grid = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)
    x_hat = fi(p_grid)
    p_grid = p_grid.reshape(64, 64, 2)
    x_hat = x_hat.reshape(64, 64)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax[0].imshow(x_hat, vmax=3, vmin=-3)
    fig.colorbar(im1)
    im2 = ax[1].imshow(x_full[0], vmax=3, vmin=-3)
    fig.colorbar(im2)
    plt.show()

if __name__ == "__main__":
    main()