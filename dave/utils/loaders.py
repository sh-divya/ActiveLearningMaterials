import os.path as osp
from copy import copy
from pathlib import Path

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

from dave.proxies.data import CrystalFeat, CrystalGraph
from dave.utils.misc import ROOT


def make_loaders(config):
    root = copy(ROOT)
    model, data = config["config"].split("-")
    if not config.get("root"):
        pass
    else:
        root = Path(osp.expandvars(config["root"])).resolve()

    if data == "mp20":
        name = "mp20"
    elif data == "mbform":
        name = "matbench_mp_e_form"
    elif data == "mbgap":
        name = "matbench_mp_e_gap"
    else:
        raise ValueError(f"Unknown config: {config['config']}")

    if model in {"fae", "faecry", "sch"}:
        load_class = GraphLoader
        trainset = CrystalGraph(
            root=str(config["root"]),
            transform=config["scales"],
            pre_transform=None,
            pre_filter=None,
            name=name,
            frame_averaging=config.get("frame_averaging"),
            fa_method=config.get("frame_averaging"),
            return_pyxtal=config.get("frame_averaging"),
            subset="train",
        )
        valset = CrystalGraph(
            root=str(config["root"]),
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
            root=config["src"].replace("$root", str(root)),
            target=config["target"],
            subset="train",
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )
        valset = CrystalFeat(
            root=config["src"].replace("$root", str(root)),
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
