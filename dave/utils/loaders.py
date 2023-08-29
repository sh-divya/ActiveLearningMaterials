from torch.utils.data import DataLoader
from dave.proxies.data import CrystalFeat
from dave.utils.misc import ROOT
from pathlib import Path
from copy import copy

def make_loaders(config):
    model, data = config["config"].split("-")
    if not config.get("root"):
        root = copy(ROOT)
    else:
        root = Path(config["root"]).resolve()

    if config["config"].endswith("-mp20"):
        name = "mp20"
        pass
    elif config["config"].endswith("-mbform"):
        name = "matbench_mp_e_form"
        config["model"]["input_len"] = 91
    elif config["config"].endswith("-mbgap"):
        config["model"]["input_len"] = 91
    else:
        raise ValueError(f"Unknown config: {config['config']}")

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
            trainset, batch_size=config["optim"]["batch_size"], shuffle=True
        ),
        "val": load_class(
            valset, batch_size=config["optim"]["batch_size"], shuffle=False
        ),
    }
