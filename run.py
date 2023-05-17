import warnings
import sys
import time
import torch

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
        # args.append("--debug")
        args.append("--config=mlp-mp20")
        args.append("--optim.epochs=1")
        #args.append("--optim.scheduler.name=ReduceLROnPlateau")
        # args.append("--optim.scheduler.name=StepLR")
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
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu") == "cuda"
    criterion = nn.MSELoss()
    module = ProxyModule(model, criterion, config, gpu)

    # Make PL trainer
    trainer = pl.Trainer(
        max_epochs=config["optim"]["epochs"],
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        min_epochs=1,
    )

    # Start training
    s = time.time()
    trainer.fit(
        model=module,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )
    t = time.time() - s

    # Inference time
    inf_s = time.time()
    trainer.test(module, loaders["val"], ckpt_path="best", verbose=False)
    inf_t = time.time() - inf_s

    # End of training
    if logger:
        logger.experiment.summary["trainer-time"] = t
        logger.experiment.summary["inference-time"] = inf_t
        logger.experiment.finish()
