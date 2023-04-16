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

from config.mp20 import config
from proxies.data import CrystalFeat
from proxies.models import ProxyMLP, ProxyModel
from utils.callbacks import get_checkpoint_callback
from utils.misc import (
    flatten_grid_search,
    get_run_dir,
    merge_dicts,
    print_config,
    resolve,
)
from utils.parser import parse_args_to_dict

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
    early = EarlyStopping(monitor="val_acc", patience=config["es_patience"], mode="min")
    ckpt = get_checkpoint_callback(
        config["run_dir"], logger, monitor="val_acc", mode="min"
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
    cli_conf = parse_args_to_dict()
    config["run_dir"] = get_run_dir()
    config = merge_dicts(config, cli_conf)
    config["run_dir"] = resolve(config["run_dir"])
    config["run_dir"].mkdir(parents=True, exist_ok=True)
    config["run_dir"] = str(config["run_dir"])

    print_config(config)

    model_config = copy.copy(config["model_config"])
    if config.get("model_grid_search"):
        grid = flatten_grid_search(config["model_grid_search"])
    else:
        grid = [{}]

    for m, model_grid_conf in enumerate(grid):
        if len(grid) > 1:
            print("\n" + "=" * 80)
            print("=" * 80)
            print_config(model_grid_conf)

        config["model_config"] = merge_dicts(model_config, model_grid_conf)
        name = "_".join(
            [f"{key}-{config['model_config'][key]}" for key in ["lr", "batch_size"]]
        )
        # or "_".join([f"{k}-{v}" for k, v in model_grid_conf.items()])

        if not config.get("no_logger"):
            logger = WandbLogger(
                project="Proxy-MP20",
                name=(name),
                entity="mila-ocp",
            )
        else:
            logger = None

        train(config, logger=logger)
