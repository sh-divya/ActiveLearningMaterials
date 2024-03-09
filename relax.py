import sys
from copy import copy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pyxtal import pyxtal
from pyxtal.lattice import Lattice

# from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.ase import Relaxer

from dave.proxies.models import make_model
from dave.proxies.data import CrystalFeat
from dave.utils.misc import load_config, set_seeds, ROOT, resolve


root = "/home/minion/Documents/materials_dataset_v3/data/matbench_mp_e_form/"
csv = "train_data.csv"


def load_model_and_weights(config):
    # with torch.device("cuda"):
    model = make_model(config)
    if not config.get("weights"):
        ValueError(
            "No model weights specified. Please provide path to model state dictionary"
        )
    else:
        state_dix = torch.load(str(config["root"] / "model.pt"))
        model.load_state_dict(state_dix)
        model.to(config["device"])
        model.eval()

    return model


def inverse_proxy_scale(pred, scale):
    return pred * scale["std"] + scale["mean"]


def proxy_predictions(loader, model, yscale, device):

    preds = []
    for batch in loader:
        x, _ = batch
        out = model(x).squeeze(-1)
        out = inverse_proxy_scale(out, yscale)
        preds.extend(out.tolist())
    return preds


def relaxed_predictions(df, num_gen, proxy_col, device):
    GetASE = AseAtomsAdaptor()
    # relax = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    # relax = Relaxer(relax)
    predict = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    all_cols = df.columns
    rel_vals = []
    rel_str = []

    for row in df.iterrows():

        comp = row[1][all_cols[7:-2]]
        form = [(comp.index[k], i) for k, i in enumerate(comp) if i > 0]
        elems, count = zip(*form)
        sg = row[1][all_cols[0]]
        lat_par = row[1][all_cols[1:7]]
        abc = lat_par[:3].to_list()
        angles = lat_par[3:].to_list()
        lattice = Lattice.from_para(*abc, *angles)

        s = pyxtal()
        minE = 1000
        for i in range(num_gen):
            s.from_random(3, sg, elems, count, lattice=lattice)
            pmg = s.to_pymatgen()
            pmg = pmg.relax(verbose=False, steps=100)
            eform = predict.predict_structure(pmg)
            delta = eform - row[1][proxy_col]
            if delta < minE:
                minE = delta
                relE = eform
                relS = pmg
        rel_vals.append(relE.cpu())
        rel_str.append(relS)

    return rel_vals, rel_str


def load_data(config):
    if not config.get("root"):
        data_root = copy(ROOT) / "dave" / "proxies"
    else:
        data_root = resolve(config["root"]) / "data"

    model, data = config["config"].split("-")
    if data == "mbform":
        name = "matbench_mp_e_form"
    else:
        raise ValueError(f"{name} not implemented for relax pipeline")

    if model in ["fae", "faecry", "sch", "pyxtal_faenet"]:
        ValueError(f"Position based predictors not implemented for relax pipeline")
    else:
        tmp_path = config["src"].replace("$root", str(data_root))
        sample_name = config.get("csv", "train")
        sampleset = CrystalFeat(
            root=tmp_path,
            target=config["target"],
            subset=sample_name,
            scalex=config["scales"]["x"],
            scaley=config["scales"]["y"],
        )
        df = pd.read_csv(str(Path(tmp_path) / f"{sample_name}_data.csv"), index_col=0)
        loader = DataLoader(
            sampleset,
            batch_size=1024,
            shuffle=False,
            pin_memory=True,
            num_workers=config["optim"].get("num_workers", 0),
        )
        config["root"] = Path(tmp_path)

    return loader, df


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = sys.argv[1:]
    set_seeds(0)
    config = load_config()

    dataloader, df = load_data(config)
    proxy_model = load_model_and_weights(config)
    target = f'{config["target"]}'
    df[f"{target}_proxy"] = proxy_predictions(
        dataloader, proxy_model, config["scales"]["y"], torch.device(device)
    )

    del proxy_model
    del dataloader

    if not config.get("num_gen"):
        config["num_gen"] = 2
    else:
        config["num_gen"] = int(config["num_gen"])

    df = df.iloc[:2]

    torch.set_default_device(device)

    vals, structs = relaxed_predictions(
        df, config["num_gen"], f"{target}_proxy", device
    )

    df[f"{target}_relax"] = vals
    df["rel_str"] = structs
