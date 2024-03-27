import os.path as osp
from copy import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

# from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.loader import DataLoader as GraphLoader

from dave.proxies.data import CrystalFeat
from dave.utils.misc import ROOT, resolve


def update_loaders(trainloader, valloader):
    num_workers = trainloader.num_workers
    batch_size = trainloader.batch_size
    tset = trainloader.dataset
    vset = valloader.dataset
    tset = ConcatDataset([tset, vset])
    vindx = list(range(len(vset)))
    vset = Subset(tset, vindx)

    trainloader = DataLoader(
        tset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    valloader = DataLoader(
        vset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return {"train": trainloader, "val": valloader}


def make_loaders(config):
    data_root = copy(ROOT)
    model, data = config["config"].split("-")
    if not config.get("root"):
        data_root = resolve(
            "/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v3"
        )
        config["root"] = data_root
        print("\nWarning, no data root specified, using default")
        print(str(data_root) + "\n")
    else:
        data_root = resolve(config["root"])

    if data == "mp20":
        name = "mp20"
    elif data == "mbform":
        name = "matbench_mp_e_form"
    elif data == "mbgap":
        name = "matbench_mp_e_gap"
    elif data == "ic":
        name = "nrcc_ionic_conductivity"
    else:
        raise ValueError(f"Unknown config: {config['config']}")

    if model in {"fae", "faecry", "sch", "pyxtal_faenet"}:
        load_class = GraphLoader
        trainset = CrystalGraph(
            root=config["root"],
            transform=config["scales"],
            pre_transform=None,
            pre_filter=None,
            name=name,
            frame_averaging=config.get("frame_averaging"),
            fa_method=config.get("fa_method"),
            return_pyxtal=config.get("return_pyxtal"),
            subset="train",
        )

    else:
        load_class = DataLoader
        trainset = CrystalFeat(
            root=config["src"].replace("$root", str(data_root)),
            target=config["target"],
            subset="train",
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )

    if config.get("crossval"):

        folds = config["crossval"]

        num_samples = len(trainset)
        split = num_samples // folds
        num_samples = num_samples - split
        split = [num_samples, split]
        trainset, valset = random_split(trainset, split)
        folds = folds - 1
        train_subsets = []
        while folds > 0:
            num_samples = num_samples - split[-1]
            split = [num_samples, split[-1]]
            trainset, tmpset = random_split(trainset, split)
            train_subsets.append(tmpset)
            folds = folds - 1
        valoader = load_class(
            valset,
            batch_size=config["optim"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=config.get("num_workers", 0),
        )
        trainset = ConcatDataset(train_subsets)
        tr_loader = load_class(
            trainset,
            batch_size=config["optim"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=config.get("num_workers", 0),
        )

        return {"train": tr_loader, "val": valoader}

    else:
        if isinstance(trainset, CrystalFeat):
            valset = CrystalFeat(
                root=config["src"].replace("$root", str(data_root)),
                target=config["target"],
                subset="val",
                scalex=config["scales"]["x"],
                scaley=config["scales"]["y"],
            )
        else:
            valset = CrystalGraph(
                root=config["root"],
                transform=config["scales"],
                pre_transform=None,
                pre_filter=None,
                name=name,
                frame_averaging=config.get("frame_averaging"),
                fa_method=config.get("fa_method"),
                return_pyxtal=config.get("return_pyxtal"),
                subset="val",
            )

    return {
        "train": DataLoader(
            trainset, batch_size=config["optim"]["batch_size"], shuffle=True
        ),
        "val": DataLoader(
            valset, batch_size=config["optim"]["batch_size"], shuffle=False
        ),
    }
