import warnings
import sys
import time

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

from dave.proxies.models import make_model
from dave.proxies.pl_modules import ProxyModule
from dave.utils.callbacks import get_checkpoint_callback
from dave.utils.loaders import make_loaders
from dave.utils.misc import load_config, print_config, set_seeds

from tqdm import tqdm
import torch
from dave.utils.misc import preprocess_data

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def custom_fit(model, train_dataloader, val_dataloader, criterion, max_epochs, optimizer, device):
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/ {max_epochs}"):
            batch = batch.to(device)
            batch, y = preprocess_data(batch, "graph")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Compute loss
            loss = criterion(outputs, y)  # Calculate your loss here
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()

        # Calculate and log average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Avg. Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)  # Calculate your validation loss here
                val_loss += loss.item()

        # Calculate and log validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} - Avg. Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    # parse command-line arguments as `--key=value` and merge with config
    # allows for nested dictionaries: `--key.subkey=value``
    # load initial config from `--config={model}-{task}`

    args = sys.argv[1:]
    if all("config" not in arg for arg in args):
        args.append("--debug")
        args.append("--config=pyxtal_faenet-mbform")
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
                config["run_dir"],
                logger,
                monitor="total_val_mae",
                mode=callbacks[0].mode,
            )
        ]

    # Make module
    criterion = nn.MSELoss()
    module = ProxyModule(model, criterion, config)

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
    # TODO: debug
    # optimizer = module.configure_optimizers()['optimizer']
    # custom_fit(model, loaders["train"], loaders["val"], criterion, config["optim"]["epochs"], optimizer, module.device)

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
