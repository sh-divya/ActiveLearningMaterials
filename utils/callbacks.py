from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import Logger

from .misc import resolve


def get_checkpoint_callback(
    run_dir: Union[Path, str],
    logger: Optional[Logger] = None,
    monitor: Optional[str] = "val_acc",
    mode: Optional[str] = "max",
) -> ModelCheckpoint:
    """
    Get a checkpoint callback to save model checkpoints.

    If the logger is available and its id is not None, the checkpoints are saved in
    ``run_dir / logger.id``

    Otherwise a random ``uuid`` is used as the checkpoint directory name.

    Returns:
        ModelCheckpoint: The checkpoint callback.
    """
    dirpath = resolve(run_dir)
    if logger and logger._id:
        ckpt_dir = logger._id
    else:
        ckpt_dir = str(uuid4()).split("-")[0]

    return ModelCheckpoint(
        dirpath=dirpath / f"checkpoints-{ckpt_dir}",
        filename="{epoch}-{step}" + f"-{{{monitor}}}" if monitor else "",
        monitor=monitor,
        mode=mode,
        verbose=True,
        save_last=True,
    )
