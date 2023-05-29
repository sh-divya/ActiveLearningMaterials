import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = Path(__file__).parent.parent
proxy_path = BASE_PATH / "proxies"
script_path = BASE_PATH / "scripts"
sys.path.append(str(proxy_path))
sys.path.append(str(script_path))

from cdvae_csv import feature_per_struc, FEATURE_KEYS
from data import CrystalFeat
from data_dist import plots_from_df

import torch
import click
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
from pymatgen.core.structure import Structure
from otdd.pytorch.distance import DatasetDistance
from verstack.stratified_continuous_split import scsplit


class DFdataset(Dataset):
    def __init__(self, dataframe, target):
        feat_cols = dataframe.columns != target
        self.x = torch.tensor(dataframe.loc[:, feat_cols].values)
        dataframe["binned"] = pd.cut(dataframe[target], 200)
        self.targets = torch.tensor(dataframe.loc[:, "binned"].values)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.x[index], self.targets[index]


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
@click.option("--write_path", default=None)
@click.option("--verbose", is_flag=True, default=False)
def split(base_path, data_select, strategy, write_path, verbose):
    db_target = {"matbench_mp_e_form": "Eform", "matbench_mp_gap": "Band Gap"}
    base_path = Path(base_path)
    data_types = {k: np.int32 for k in FEATURE_KEYS}
    data_types = {k: np.float32 for k in ["a", "b", "c", "alpha", "beta", "gamma"]}

    for d in data_select:
        db = list(db_target.keys())[int(d)]
        target = db_target[db]
        read_path = base_path / db / "data" / (db + ".csv")
        df = pd.read_csv(read_path, dtype=data_types, index_col=0)
        if not write_path:
            write_path = Path(base_path / db)
        else:
            write_path = Path(write_path)
        if strategy == "stratify":
            train, val, test = proportional(df, target, verbose)
            train.to_csv(write_path / "train_data.csv")
            val.to_csv(write_path / "val_data.csv")
            test.to_csv(write_path / "test_data.csv")
            fig, ax = plt.subplots(2, 4, figsize=(15, 6))
            for s, n in zip((train, val, test), ("train", "val", "test")):
                lines, labels = plots_from_df(s, target, ax, n)
        elif strategy == "ood":
            train, id_val, od_val, test = ood(df, target, verbose)
            train.to_csv(write_path / "train_data.csv")
            id_val.to_csv(write_path / "idval_data.csv")
            od_val.to_csv(write_path / "odval_data.csv")
            test.to_csv(write_path / "test_data.csv")
            fig, ax = plt.subplots(2, 4, figsize=(15, 6))
            for s, n in zip(
                (train, id_val, od_val, test), ("train", "id_val", "od_val", "test")
            ):
                lines, labels = plots_from_df(s, target, ax, n)
        fig.legend(lines, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(write_path / (f"{d}_{strategy}.png"))
        plt.close()


def proportional(df, target, verbose):
    train_val, test = scsplit(df, stratify=df[target], test_size=0.2, continuous=True)
    train, val = scsplit(
        train_val.reset_index(drop=True),
        stratify=train_val.reset_index(drop=True)[target],
        test_size=0.25,
        continuous=True,
    )
    if verbose:
        trainds = DFdataset(train, target)
        valds = DFdataset(val, target)
        testds = DFdataset(test, target)
        print(trainds[0])
        print("Strategy: Stratifed split b/w Train, val and test")
        print("Split ratio: Train=0.6, Val=0.2, Test=0.2")
        d = DatasetDistance(
            trainds,
            valds,
            ignore_source_labels=False,
            ignore_target_labels=False,
            inner_ot_method="exact",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
        )
        dist = d.distance(maxsamples=1000)
        print("Train-Val OTDD:{dist}")
        d = DatasetDistance(
            trainds,
            testds,
            inner_ot_method="exact",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
        )
        dist = d.distance(maxsamples=1000)
        print("Train-Test OTDD:{dist}")

    return train, val, test


def ood(df, target, verbose):
    id_train, id_test = scsplit(df, stratify=df[target], test_size=0.3, continuous=True)
    train, od_test = split_from_swaps(id_train, id_test, target, 1000, 100, 100)
    train, id_val = scsplit(
        train.reset_index(drop=True),
        stratify=train.reset_index(drop=True)[target],
        test_size=0.14,
        continuous=True,
    )
    od_test, od_val = scsplit(
        od_test.reset_index(drop=True),
        stratify=od_test.reset_index(drop=True)[target],
        test_size=0.33,
        continuous=True,
    )

    if verbose:
        trainds = DFdataset(train, target)
        id_valds = DFdataset(id_val, target)
        od_valds = DFdataset(od_val, target)
        testds = DFdataset(od_test, target)
        print("Strategy: OOD split b/w Train, val and test")
        print("Split ratio: Train=0.6, Val=0.2, Test=0.2")
        d = DatasetDistance(
            trainds,
            id_valds,
            inner_ot_method="exact",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
        )
        dist = d.distance()
        print("Train-Val-ID OTDD:{dist}")
        d = DatasetDistance(
            trainds,
            od_valds,
            inner_ot_method="exact",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
        )
        dist = d.distance(maxsamples=1000)
        print("Train-Val-OD OTDD:{dist}")
        d = DatasetDistance(
            trainds,
            testds,
            inner_ot_method="exact",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
        )
        dist = d.distance(maxsamples=1000)
        print("Train-Test OTDD:{dist}")

    return train, id_val, od_val, od_test


def split_from_swaps(train, test, target, n_swaps=100, swaps_per_iter=5, history=20):
    """
    Function adapted from
    https://github.com/Confusezius/Characterizing_Generalization_in_DeepMetricLearning
    """
    train_hist, test_hist = [], []
    feat_cols = train.columns != target

    for i in range(n_swaps):
        trainmean = train.loc[:, feat_cols].mean().values.reshape(1, -1)
        testmean = test.loc[:, feat_cols].mean().values.reshape(1, -1)
        dists_train_trainmean = pairwise_distances(
            train.loc[:, feat_cols], trainmean, metric="euclidean"
        )
        dists_train_testmean = pairwise_distances(
            train.loc[:, feat_cols], testmean, metric="euclidean"
        )
        dists_test_trainmean = pairwise_distances(
            test.loc[:, feat_cols], trainmean, metric="euclidean"
        )
        dists_test_testmean = pairwise_distances(
            test.loc[:, feat_cols], testmean, metric="euclidean"
        )

        train_swaps = np.argsort(
            (dists_train_testmean - dists_train_trainmean).reshape(1, -1)
        ).squeeze(0)
        test_swaps = np.argsort(
            (dists_test_trainmean - dists_test_testmean).reshape(1, -1)
        ).squeeze(0)

        for j in range(swaps_per_iter):
            swapped_train, swapped_test = False, False
            for train_swap_temp, test_swap_temp in zip(train_swaps[j:], test_swaps[j:]):
                if train_swap_temp not in train_hist[-history:] and not swapped_train:
                    train_hist.append(train_swap_temp)
                    train_swap = train_swap_temp
                    swapped_train = True
                if test_swap_temp not in test_hist[-history:] and not swapped_test:
                    test_hist.append(test_swap_temp)
                    test_swap = test_swap_temp
                    swapped_test = True
                if swapped_train and swapped_test:
                    break

            train.iloc[train_swap, :], test.iloc[test_swap, :] = (
                test.iloc[test_swap, :],
                train.iloc[train_swap, :],
            )
    return train, test


if __name__ == "__main__":
    # write_dataset_csv()
    split()
