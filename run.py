import copy
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader

from proxies.data import CrystalFeat
from proxies.models import ProxyMLP, ProxyModel
from utils.callbacks import get_checkpoint_callback
from utils.misc import (
    print_config,
    load_config,
)

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)


def train(config, logger):
    device = torch.device("cpu")
    model_config = config["model_config"]

    trainset = CrystalFeat(
        root=model_config["root"],
        target="formation_energy_per_atom",
        subset="train",
        scalex=config["xscale"],
        scaley=config["yscale"],
    )
    valset = CrystalFeat(
        root=model_config["root"],
        target=config["target"],
        subset="val",
        scalex=config["xscale"],
        scaley=config["yscale"],
    )

    trainloader = DataLoader(
        trainset, batch_size=model_config["batch_size"], shuffle=True
    )
    valloader = DataLoader(valset, batch_size=model_config["batch_size"], shuffle=False)

    model = (
        ProxyMLP(model_config["input_len"], model_config["hidden_layers"])
        .to(device)
        .to(torch.float32)
    )
    model.apply(weights_init)
    criterion = nn.MSELoss()
    accuracy = nn.L1Loss()
    early = EarlyStopping(monitor="val_acc", patience=config["es_patience"], mode="max")
    ckpt = get_checkpoint_callback(
        config["run_dir"], logger, monitor="val_acc", mode="max"
    )

    model = ProxyModel(model, criterion, accuracy, model_config["lr"])
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ckpt, early],
        min_epochs=1,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
    if logger:
        logger.experiment.config["LR"] = model_config["lr"]
        logger.experiment.config["batch"] = model_config["batch_size"]
        logger.experiment.config["layers"] = model_config["hidden_layers"]
        logger.experiment.finish()


if __name__ == "__main__":
    # parse command-line arguments as `--key=value` and merge with config
    # allows for nested dictionaries: `--key.subkey=value``
    # load initial config from `--config={task}-{model}`
    config = load_config()
    if not config.get("wandb_run_name"):
        wandb_name_keys = {
            "model": ["hidden_layers"],
            "optim": ["lr", "batch_size"],
        }
        config["wandb_run_name"] = (
            config["config"]
            + "-"
            + "-".join(
                [
                    f"{key}={config[level][key]}"
                    for level in wandb_name_keys
                    for key in wandb_name_keys[level]
                    if key in config[level]
                ]
            )
        )
        # mp20-mlp-hidden_layers=[512, 512]-lr=0.001-batch_size=32

    print_config(config)
    if not config.get("debug"):
        logger = WandbLogger(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            entity=config["wandb_entity"],
        )
    else:
        logger = None
        print(
            "\n🛑Debug mode: run dir was not created, checkpoints"
            + " will not be saved, and no logger will be used"
        )

    # train(config, logger=logger)
