import torch
import wandb

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from typing import Dict


# Local imports
from model import Model


def train(params:Dict):
  
  # Pass your defaults to wandb.init
  wandb.init(config=params)
  config = dict(wandb.config)

  # Adjust noise magnitude
  if config["asym_noise"]:
    config["noise_rate"] /= 2.0

  pl.seed_everything(config["random_seed"])
  np.random.seed(config["random_seed"])

  model   = Model(config)

  trainer = pl.Trainer(
    gpus=1,
    max_epochs=config["max_epochs"],
    logger=[WandbLogger()]
  )

  trainer.fit(model)

  wandb.finish()

  return model



if __name__ == "__main__":

  default_params = {
    "random_seed":          1,
    "dataset":              "CIFAR-10",
    "complex_model":        False,

    "batch_size":           128,
    "valid_batch_size":     1000,
    "train_ratio":          0.9,

    "max_epochs":           60,
    "learning_rate":        0.01,
    "lr_scheduler":         "stepped",
    "lr_milestone":         20,
    "conv_l2_reg":          1e-4,
    "fc_l2_reg":            0.01,

    "asym_noise":           False,
    "noise_rate":           0.2,

    "loss_function":        "classical",

    "lambda_1":             1.0,
    "lambda_2":             0.0,
    "lambda_3":             0.0,
    "lambda_4":             0.0,
    "moments_alpha":        0.01,
    "normalize":            False,

    "wandb_log_hist":       False,
  }

  train(default_params)
