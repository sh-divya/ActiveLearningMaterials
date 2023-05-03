import warnings
import sys

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from proxies.models import make_model
from proxies.pl_modules import ProxyModule
from utils.callbacks import get_checkpoint_callback
from utils.loaders import make_loaders
from utils.misc import load_config, print_config, set_seeds

warnings.filterwarnings("ignore", ".*does not have many workers.*")


if __name__ == "__main__":
    # parse command-line arguments as `--key=value` and merge with config
    # allows for nested dictionaries: `--key.subkey=value``
    # load initial config from `--config={model}-{task}`

    args = sys.argv[1:]
    if all("config" not in arg for arg in args):
        args.append("--debug")
        args.append("--config=physmlp-mp20")
        sys.argv[1:] = args

    set_seeds(0)
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
            notes=config["wandb_note"],
        )
    else:
        logger = None
        print(
            "\n🛑Debug mode: run dir was not created, checkpoints"
            + " will not be saved, and no logger will be used"
        )

    # create dataloaders and model
    loaders = make_loaders(config)
    model = make_model(config)

    # setup PL callbacks
    callbacks = []
    callbacks += [
        EarlyStopping(
            monitor="val_acc", patience=config["optim"]["es_patience"], mode="min"
        )
    ]
    if not config.get("debug"):
        callbacks += [
            get_checkpoint_callback(
                config["run_dir"], logger, monitor="val_acc", mode=callbacks[0].mode
            )
        ]

    # Make module
    criterion = nn.MSELoss()
    accuracy = nn.L1Loss()
    module = ProxyModule(model, criterion, accuracy, config)

    # Make PL trainer
    trainer = pl.Trainer(
        max_epochs=config["optim"]["epochs"],
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        min_epochs=1,
    )

    # Start training
    trainer.fit(
        model=module,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    # End of training
    if logger:
        logger.experiment.config["lr"] = config["optim"]["lr"]
        logger.experiment.config["batch"] = config["optim"]["batch_size"]
        logger.experiment.config["layers"] = config["model"]["hidden_layers"]
        logger.experiment.finish()
