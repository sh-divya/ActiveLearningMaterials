from torch.utils.data import DataLoader

from dave.proxies.data import CrystalFeat
from dave.utils.misc import ROOT


def make_loaders(config):
    if config["config"].endswith("-mp20"):
        trainset = CrystalFeat(
            root=config["src"].replace("$root", str(ROOT)),
            target="formation_energy_per_atom",
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

    raise ValueError(f"Unknown config: {config['config']}")
