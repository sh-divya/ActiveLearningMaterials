from torch.utils.data import DataLoader
from copy import copy

from dave.proxies.data import CrystalFeat
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
