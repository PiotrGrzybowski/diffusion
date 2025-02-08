# from itertools import product
# from pathlib import Path
#
# import pytest
# from diffusion.scripts.train import train
# from hydra import compose, initialize_config_dir
# from hydra.core.hydra_config import HydraConfig
# from omegaconf import DictConfig, open_dict
#
#
# losses = ["mean_mse", "noise_mse", "mean_simple_mse", "noise_simple_mse", "variational_bound"]
# models = ["small_unet"]
#
#
# @pytest.mark.parametrize("loss, model", product(losses, models))
# def test_train_fast_dev_run(cfg_train: DictConfig, configs_path: Path, loss: str, model) -> None:
#     with initialize_config_dir(config_dir=str(configs_path), version_base="1.3"):
#         model_config = compose(config_name=f"model/{model}")
#         loss_config = compose(config_name=f"loss/{loss}")
#
#         with open_dict(cfg_train):
#             cfg_train.model = model_config.model
#             cfg_train.loss = loss_config.loss
#
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.trainer.fast_dev_run = True
#         cfg_train.trainer.accelerator = "cpu"
#         cfg_train.logger = None
#         cfg_train.callbacks = None
#         cfg_train.test = False
#     train(cfg_train)
