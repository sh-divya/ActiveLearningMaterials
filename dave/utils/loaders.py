from torch.utils.data import DataLoader

from dave.proxies.data import CrystalFeat
from dave.utils.misc import ROOT


def make_loaders(config):
    if config["config"].endswith("-mp20"):
        pass
    elif config["config"].endswith("-mbform"):
        config["model"]["input_len"] = 91
    elif config["config"].endswith("-mbgap"):
        config["model"]["input_len"] = 91
    else:
        raise ValueError(f"Unknown config: {config['config']}")

    trainset = CrystalFeat(
        root=config["src"].replace("$root", str(ROOT)),
        target=config["target"],
        subset="train",
        scalex=config["scales"]["x"],
        scaley=config["scales"]["y"],
    )
    valset = CrystalFeat(
        root=config["src"].replace("$root", str(ROOT)),
        target=config["target"],
        subset="val",
        scalex=config["scales"]["x"],
        scaley=config["scales"]["y"],
    )

    return {
        "train": DataLoader(
            trainset, batch_size=config["optim"]["batch_size"], shuffle=True
        ),
        "val": DataLoader(
            valset, batch_size=config["optim"]["batch_size"], shuffle=False
        ),
    }
