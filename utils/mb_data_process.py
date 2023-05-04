import os
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cdvae_csv import feature_per_struc, FEATURE_KEYS

import click
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from verstack.stratified_continuous_split import scsplit



@click.command()
@click.option("--read_path", default="./data")
@click.option("--write_base", default="./proxies")
def write_dataset_csv(read_path, write_base):
    db_path = Path(read_path)
    write_base = Path(write_base)
    db_name = ["matbench_mp_e_form", "matbench_mp_gap"]
    targets = ["Eform", "Band Gap"]

    for i, db in enumerate(db_name):
        y = []
        proxy_features = {key: [] for key in FEATURE_KEYS}
        json_file = db_path / (db + ".json")
        with open(json_file, "r") as fobj:
            jobj = json.load(fobj)
            for j in jobj["data"]:
                struc = Structure.from_dict(j[0])
                proxy_features = feature_per_struc(struc, proxy_features)
                y.append(j[1])
        proxy_features[targets[i]] = y
        df = pd.DataFrame.from_dict(proxy_features)
        df = df.loc[:, (df != 0).any()]
        df.to_csv(write_base / db / "data" / (db + ".csv"))


@click.command()
@click.option("--base_path", default="./proxies")
def stratify_split_dataset(base_path):
    db_name = ["matbench_mp_e_form", "matbench_mp_gap"]
    targets = ["Eform", "Band Gap"]
    base_path = Path(base_path)
    data_types = {k: np.int32 for k in FEATURE_KEYS}
    data_types = {k: np.float32 for k in ["a", "b", "c", "alpha", "beta", "gamma"]}
    for i, db in enumerate(db_name):
        read_path = base_path / db / "data" / (db + ".csv")
        df = pd.read_csv(read_path, dtype=data_types, index_col=0)
        train_val, test = scsplit(
            df, stratify=df[targets[i]], test_size=0.2, continuous=True
        )
        train, val = scsplit(
            train_val.reset_index(drop=True),
            stratify=train_val.reset_index(drop=True)[targets[i]],
            test_size=0.25,
            continuous=True,
        )
        train.to_csv(base_path / db / "train.csv")
        val.to_csv(base_path / db / "val.csv")
        test.to_csv(base_path / db / "test.csv")


if __name__ == "__main__":
    # write_dataset_csv()
    stratify_split_dataset()
