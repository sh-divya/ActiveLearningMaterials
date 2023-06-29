from torch.utils.data import DataLoader

from dave.proxies.data import CrystalFeat, MatBenchDataset
from dave.utils.misc import ROOT, load_matbench_train_val_indices


def make_loaders(config, mb_data=None):
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
                trainset,
                batch_size=config["optim"]["batch_size"],
                num_workers=config["optim"]["num_workers"],
                shuffle=True,
                pin_memory=True,
            ),
            "val": DataLoader(
                valset,
                batch_size=config["optim"]["batch_size"],
                num_workers=config["optim"]["num_workers"],
                shuffle=False,
                pin_memory=True,
            ),
        }
    elif config["config"].endswith("-matbench"):
        train_indices, val_indices = load_matbench_train_val_indices(
            config["fold"], config["val_frac"]
        )

        trainset = MatBenchDataset(
            config["src"],
            train_indices,
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )

        valset = MatBenchDataset(
            config["src"],
            val_indices,
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )

        return {
            "train": DataLoader(
                trainset,
                batch_size=config["optim"]["batch_size"],
                num_workers=config["optim"]["num_workers"],
                shuffle=True,
                pin_memory=True,
            ),
            "val": DataLoader(
                valset,
                batch_size=config["optim"]["batch_size"],
                num_workers=config["optim"]["num_workers"],
                shuffle=False,
                pin_memory=True,
            ),
        }

    raise ValueError(f"Unknown config: {config['config']}")
