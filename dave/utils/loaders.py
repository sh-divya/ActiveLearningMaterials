import os.path as osp
from copy import copy
from pathlib import Path

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

from dave.proxies.data import CrystalFeat, CrystalGraph
from dave.utils.misc import ROOT, resolve


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
    else:
        load_class = DataLoader
        trainset = CrystalFeat(
            root=config["src"].replace("$root", str(data_root)),
            target=config["target"],
            subset="train",
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )
        valset = CrystalFeat(
            root=config["src"].replace("$root", str(data_root)),
            target=config["target"],
            subset="val",
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )

    return {
        "train": load_class(
            trainset,
            batch_size=config["optim"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=config["optim"].get("num_workers", 0),
        ),
        "val": load_class(
            valset,
            batch_size=config["optim"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=config["optim"].get("num_workers", 0),
        ),
    }
