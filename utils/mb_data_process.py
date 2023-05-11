import os
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from cdvae_csv import feature_per_struc, FEATURE_KEYS

import click
import scipy as sp
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
@click.option("--data_select", default="01")
@click.option("--strategy", default="stratify")
def split(base_path, data_select, strategy):
    db_target = {"matbench_mp_e_form": "Eform", "matbench_mp_gap": "Band Gap"}
    base_path = Path(base_path)
    data_types = {k: np.int32 for k in FEATURE_KEYS}
    data_types = {k: np.float32 for k in ["a", "b", "c", "alpha", "beta", "gamma"]}
    for d in data_select:
        db = list(db_target.keys())[int(d)]
        target = db_target[db]
        read_path = base_path / db / "data" / (db + ".csv")
        df = pd.read_csv(read_path, dtype=data_types, index_col=0)
        if strategy == "stratify":
            train, val, test = proportional(df, target)
            train.to_csv(base_path / db / "train.csv")
            val.to_csv(base_path / db / "val.csv")
            test.to_csv(base_path / db / "test.csv")
        elif strategy == "ood":
            temp = ood(df, target)
            print(temp)


def divergence(ptarget, qtarget, bins):
    p = pd.cut(ptarget, bins=bins).value_counts(normalize=True)
    q = pd.cut(qtarget, bins=bins).value_counts(normalize=True)
    return sp.stats.entropy(p, q)


def proportional(df, target):
    train_val, test = scsplit(df, stratify=df[target], test_size=0.2, continuous=True)
    train, val = scsplit(
        train_val.reset_index(drop=True),
        stratify=train_val.reset_index(drop=True)[target],
        test_size=0.25,
        continuous=True,
    )

    return train, val, test


def ood(df, target):
    min_t, max_t = df[target].min(), df[target].max()
    step = (max_t - min_t) / 200
    bins = np.linspace(min_t - step, max_t + step, 200)
    id_train, id_test = scsplit(df, stratify=df[target], test_size=0.2, continuous=True)
    div = divergence(id_train[target], id_test[target], bins)
    return div


if __name__ == "__main__":
    # write_dataset_csv()
    split()
