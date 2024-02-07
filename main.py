import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dynabench.dataset.download import download_equation

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    '''
    GrIND -> GRid Interpolation Network for Dynamical systems
    '''
    datamodule = instantiate(cfg.datamodule)

    model = instantiate(cfg.model)
    
    trainer = instantiate(cfg.trainer)
    
    #trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="output/tb_logs/lightning_logs/advection-high-cloud-test-euler/checkpoints/epoch=0-step=10000.ckpt")

if __name__ == "__main__":
    main()