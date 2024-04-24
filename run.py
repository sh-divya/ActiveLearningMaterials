import sys
import time
import warnings

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

from dave.proxies.models import make_model
from dave.proxies.pl_modules import ProxyModule
from dave.utils.callbacks import get_checkpoint_callback
from dave.utils.loaders import make_loaders, update_loaders
from dave.utils.misc import load_config, print_config, set_seeds

warnings.filterwarnings("ignore", ".*does not have many workers.*")


if __name__ == "__main__":
    # parse command-line arguments as `--key=value` and merge with config
    # allows for nested dictionaries: `--key.subkey=value``
    # load initial config from `--config={model}-{task}`

    args = sys.argv[1:]
    if all("config" not in arg for arg in args):
        # args.append("--debug")
        args.append("--config=mlp-mp20")
        # args.append("--optim.scheduler.name=StepLR")
        warnings.warn("No config file specified, using default !")
        sys.argv[1:] = args

    set_seeds(0)
    config = load_config()
    if not config.get("wandb_run_name"):
        config["wandb_run_name"] = config["run_dir"].split("/")[-1]

    print_config(config)
    if not config.get("debug"):
        logger = WandbLogger(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            entity=config["wandb_entity"],
            notes=config["wandb_note"],
            tags=config["wandb_tags"],
        )
    else:
        logger = DummyLogger()
        print(
            "\n🛑Debug mode: run dir was not created, checkpoints"
            + " will not be saved, and no logger will be used\n"
        )

    # create dataloaders and model
    loaders = make_loaders(config)
    model = make_model(config)
    crossval = int(config.get("crossval", 1))

    # setup PL callbacks
    callbacks = []
    callbacks += [
        EarlyStopping(
            monitor="val_mae", patience=config["optim"]["es_patience"], mode="min"
        )
    ]
    if not config.get("debug"):
        callbacks += [
            get_checkpoint_callback(
                config["run_dir"], logger, monitor="val_mae", mode=callbacks[0].mode
            )
        ]

    # Make module
    criterion = nn.MSELoss()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    module = ProxyModule(model, criterion, config)  # .to(device)
    epochs = config["optim"]["epochs"] if crossval == 1 else 1

    # Make PL trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        min_epochs=1,
    )

    # Start training
    s = time.time()

    if epochs > 1:
        trainer.fit(
            model=module,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )
    else:
        epochs = config["optim"]["epochs"]
        for _ in range(epochs):
            trainer.fit(
                model=module,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
            )
            if crossval > 1:
                trainer.fit_loop.epoch_progress.reset_on_run()
                loaders = update_loaders(loaders["train"], loaders["val"])

    t = time.time() - s

    # Inference time
    inf_s = time.time()
    trainer.test(module, loaders["val"], ckpt_path="best", verbose=False)
    inf_t = time.time() - inf_s

    # End of training
    if logger:
        if isinstance(logger.experiment.summary, dict):
            logger.experiment.summary["trainer-time"] = t
            logger.experiment.summary["inference-time"] = inf_t
        logger.experiment.finish()
