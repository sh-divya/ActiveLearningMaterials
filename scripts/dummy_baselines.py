from pathlib import Path
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.mp20 import config
from dave.proxies.data import CrystalFeat


if __name__ == "__main__":
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

    losses = {
        split: {
            loss: {"zero": [], "rand": [], "mean": [], "normal": [], "unit_normal": []}
            for loss in ["mae", "mse"]
        }
        for split in ["train", "val"]
    }
    loaders = {"train": trainloader, "val": valloader}
    mae = nn.L1Loss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    loss_funcs = {"mae": mae, "mse": mse}

    for split, loader in loaders.items():
        for _, y in loader:
            for loss_name, loss_func in loss_funcs.items():
                losses[split][loss_name]["zero"].append(
                    loss_func(torch.zeros_like(y), y)
                )
                losses[split][loss_name]["rand"].append(
                    loss_func(torch.rand_like(y), y)
                )
                losses[split][loss_name]["mean"].append(
                    loss_func(torch.full_like(y, y.mean()), y)
                )
                losses[split][loss_name]["unit_normal"].append(
                    loss_func(torch.normal(0, 1, size=y.shape), y)
                )
                losses[split][loss_name]["normal"].append(
                    loss_func(torch.normal(y.mean(), y.std(), size=y.shape), y)
                )

    for split in losses:
        for loss in losses[split]:
            for baseline in losses[split][loss]:
                v = torch.cat(losses[split][loss][baseline]).mean().item()
                print(f"{split:<6} {loss} {baseline:<12}: {v:.5f}")
