import copy
import random

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from proxies.data import CrystalFeat
from proxies.models import ProxyMLP, ProxyModel
from config.mp20 import config

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
    early = EarlyStopping(monitor="val_acc", patience=3, mode="max")

    model = ProxyModel(model, criterion, accuracy, model_config["lr"])
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        log_every_n_steps=1,
        callbacks=early,
        min_epochs=1,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
    if logger:
        logger.experiment.config["LR"] = model_config["lr"]
        logger.experiment.config["batch"] = model_config["batch_size"]
        logger.experiment.config["layers"] = model_config["hidden_layers"]
        logger.experiment.finish()


def main(config):
    model_config = copy.copy(config["model_config"])
    tune_var = config["tune_var"]

    for var in model_config[tune_var]:
        config["model_config"][tune_var] = var
        name = [
            key + "-" + str(config["model_config"][key]) for key in ["lr", "batch_size"]
        ]
        # logger = None
        logger = WandbLogger(project="Proxy-MP20", name="_".join(name))
        train(config, logger=logger)


if __name__ == "__main__":
    main(config)
