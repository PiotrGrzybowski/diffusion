from diffusion.train import train
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from rootutils import rootutils


configs_path = rootutils.find_root("pyproject.toml") / "configs"


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    with initialize_config_dir(config_dir=str(configs_path)):
        model_config = compose(config_name="model/little")

        with open_dict(cfg_train):
            cfg_train.model = model_config.model

    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.test = False
    train(cfg_train)
