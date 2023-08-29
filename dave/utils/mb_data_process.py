import os
import sys
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = Path(__file__).parent.parent
script_path = BASE_PATH.parent / "scripts"
sys.path.append(str(script_path))

from dave.utils.cdvae_csv import feature_per_struc, FEATURE_KEYS
from data_dist import plots_from_df

import torch
import click
import scipy as sp
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
from pymatgen.core.structure import Structure

from otdd.pytorch.distance import DatasetDistance
from verstack.stratified_continuous_split import scsplit
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


SG_DIX = np.load(BASE_PATH / "utils" / "sg_decomp.npy")


class DFdataset(Dataset):
    def __init__(self, dataframe, target):
        feat_cols = dataframe.columns != target
        self.x = torch.tensor(dataframe.loc[:, feat_cols].values)
        min_t, max_t = dataframe[target].min(), dataframe[target].max()
        step = (max_t - min_t) / 200
        bins = np.linspace(min_t - step, max_t + step, 200)
        labels = list(range(1, 200))
        dataframe["binned"] = pd.cut(dataframe[target], bins=bins, labels=labels)
        self.targets = torch.tensor(dataframe.loc[:, "binned"].values)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.x[index], self.targets[index]


@click.command()
@click.option("--read_path", default="./data")
@click.option("--write_base", default="./dave/proxies")
@click.option("--data", default="0")
def write_dataset_csv(read_path, write_base, data):
    """
    Reads JSON files and extracts relevant features.

    Args:
        read_path (str): Path to the directory containing JSON files.
        Default is "./data".
        write_base (str): Base directory for writing output CSV files.
        Default is "./dave/proxies".
        data (str): Data source identifier. Default is "0".

    Directory Structure:
        - The JSON files should be located in the `read_path` directory.
        - The JSON file names should correspond to the values in the `db_name` list,
          with the `.json` extension.
        - The output CSV files will be written in the `write_base` directory.
        - Within the `write_base` directory, there will be subdirectories for each
          database name in the `db_name` list.
        - Within each database subdirectory, there will be a subdirectory named `data`,
          where the output CSV files will be written.
        - The output CSV file names will be the same as the respective database names,
          but with the `.csv` extension.

        {read_path}/
            matbench_mp_e_form.json
            matbench_mp_gap.json

        {write_base}/
            {db_name[0]}/
                data/
                    {db_name[0]}.csv

            {db_name[1]}/
                data/
                    {db_name[1]}.csv

    Returns:
        None
    """
    db_path = Path(read_path)
    write_base = Path(write_base)
    db_name = ["matbench_mp_e_form", "matbench_mp_gap"]
    targets = ["Eform", "Band Gap"]

    for i, db in enumerate(db_name):
        if i in set([int(d) for d in data]):
            y = []
            proxy_features = {key: [] for key in FEATURE_KEYS}
            cif_str = []
            mb_ids = []
            json_file = db_path / (db + ".json")
            with open(json_file, "r") as fobj:
                jobj = json.load(fobj)
                for j in jobj["data"]:
                    struc = Structure.from_dict(j[0])
                    SGA = SpacegroupAnalyzer(struc)
                    struc = SGA.get_conventional_standard_structure()
                    proxy_features = feature_per_struc(struc, proxy_features)
                    cif_str.append(struc.to(None, fmt="cif"))
                    y.append(j[1])
            proxy_features[targets[i]] = y
            proxy_features["cif"] = cif_str
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
    """
    Splits data into training, validation, and test sets based on specified strategy.

    Args:
        base_path (str): Base directory containing the input CSV files. Default is "./proxies".
        data_select (str): Selection identifier for the dataset. Default is "01".
        strategy (str): Strategy for data splitting. Default is "stratify".
        write_path (str): Directory path to write the output files. If not provided, the base_path is used.
        verbose (bool): Whether to enable verbose output. Default is False.

    Directory Structure:
        - The input CSV files should be located in the `base_path` directory.
        - The input CSV file names should correspond to the database names in the `db_target` dictionary.
        - The output files will be written in the `write_path` directory.
        - If `write_path` is not provided, the base_path will be used for writing the output files.
        - Within the `write_path` directory, there will be subdirectories corresponding to each database.
        - The subdirectories will be named based on the respective database names.
        - The output files will have different names and extensions depending on the selected strategy.

        {base_path}/
            {db_name[0]}/
                data/
                    {db_name[0]}.csv
                train_data.csv
                val_data.csv
                test_data.csv
                {data_select[0]}_{strategy}.png

            {db_name[1]}/
                data/
                    {db_name[1]}.csv
                train_data.csv
                val_data.csv
                test_data.csv
                {data_select[1]}_{strategy}.png


    Returns:
        None
    """
    db_target = {"matbench_mp_e_form": "Eform", "matbench_mp_gap": "Band Gap"}
    base_path = Path(base_path)
    data_types = {k: np.int32 for k in FEATURE_KEYS}
    data_types = {k: np.float32 for k in ["a", "b", "c", "alpha", "beta", "gamma"]}
    if not write_path:
        write_path = Path(base_path)
    else:
        write_path = Path(write_path)
    for d in data_select:
        db = list(db_target.keys())[int(d)]
        target = db_target[db]
        read_path = base_path / db / "data" / (db + ".csv")
        df = pd.read_csv(read_path, dtype=data_types, index_col=0)
        if (write_path / db).is_dir():
            db_write = write_path / db
        else:
            db_write = write_path
        if strategy == "stratify":
            train, val, test = proportional(df, target, verbose)
            train.to_csv(db_write / "train_data.csv")
            val.to_csv(db_write / "val_data.csv")
            test.to_csv(db_write / "test_data.csv")
            fig, ax = plt.subplots(2, 4, figsize=(15, 6))
            for s, n in zip((train, val, test), ("train", "val", "test")):
                lines, labels = plots_from_df(s, target, ax, n)
        elif strategy == "ood":
            train, id_val, od_val, test = ood(df, target, verbose)
            train.to_csv(db_write / f"train_data_{strategy}.csv")
            id_val.to_csv(db_write / "idval_data.csv")
            od_val.to_csv(db_write / "odval_data.csv")
            test.to_csv(db_write / f"test_data_{strategy}.csv")
            fig, ax = plt.subplots(2, 4, figsize=(15, 6))
            for s, n in zip(
                (train, id_val, od_val, test), ("train", "id_val", "od_val", "test")
            ):
                lines, labels = plots_from_df(s, target, ax, n)
        fig.legend(lines, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(db_write / (f"{d}_{strategy}.png"))
        plt.close()


def comp_string(comp, pt):
    compound = {}
    for e, element in enumerate(pt):
        if comp[e] > 0:
            compound[element] = comp[e]
    return compound


def jaccard(comp1, comp2):
    idx1 = comp1 > 0
    idx2 = comp2 > 0
    idx_inter = [i and j for i, j in zip(idx1, idx2)]
    idx_union = [i or j for i, j in zip(idx1, idx2)]

    anb = sum(map(min, zip(comp1[idx_inter], comp2[idx_inter])))
    aub = sum(map(sum, zip(comp1[idx_union], comp2[idx_union])))

    return 1 - anb / aub


def levenshtein(comp1, comp2):
    """
    Levenshtein distance as in
    https://python-course.eu/applications-python/levenshtein-distance.php
    """
    comp1 = "".join([f"{k}{v}" for k, v in comp1.items()])
    comp2 = "".join([f"{k}{v}" for k, v in comp2.items()])

    rows = len(comp1) + 1
    cols = len(comp2) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if comp1[row - 1] == comp2[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,
                dist[row][col - 1] + 1,
                dist[row - 1][col - 1] + cost,
            )
    return dist[row][col]


def custom_levenshtein(comp1, comp2):
    """Calculate the Levenshtein distance between two compound dicts.

    This is adapted from pyenchant
    """
    el1 = list(comp1.keys())
    el2 = list(comp2.keys())
    all_els = FEATURE_KEYS[7:]

    rows = len(el1) + 1
    cols = len(el2) + 1

    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i

    for j in range(1, cols):
        for i in range(1, rows):
            z1 = all_els.index(el1[i - 1]) + 1
            z2 = all_els.index(el2[j - 1]) + 1
            if el1[i - 1] == el2[j - 1]:
                cost = abs(comp1[el1[i - 1]] - comp2[el1[j - 1]]) * z1
            else:
                cost = abs(comp1[el1[i - 1]] * z1 - comp2[el2[j - 1]] * z2)
            dist[i][j] = min(
                dist[i - 1][j] + z1 * comp1[el1[i - 1]],
                dist[i][j - 1] + z2 * comp2[el2[j - 1]],
                dist[i - 1][j - 1] + cost,
            )
    return dist[i][j]


def custom_distance(s1, s2, elements):
    comp1 = comp_string(s1[7:], elements)
    comp2 = comp_string(s2[7:], elements)
    abc_dist = np.linalg.norm(s1[1:4] - s2[1:4])
    angles_dist = np.linalg.norm(s1[4:7] - s1[4:7])
    sg_dist = sklearn.metrics.jaccard_score(
        SG_DIX[int(s1[0]) - 1], SG_DIX[int(s2[0]) - 1]
    )
    # return sg_dist + abc_dist + angles_dist + levenshtein(comp1, comp2)
    return sg_dist + abc_dist + angles_dist + jaccard(s1[7:], s2[7:])


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
        print("Strategy: Stratifed split b/w Train, val and test")
        print("Split ratio: Train=0.6, Val=0.2, Test=0.2")
        d = DatasetDistance(
            trainds,
            valds,
            ignore_source_labels=True,
            ignore_target_labels=True,
            inner_ot_method="gaussian_approx",
            debiased_loss=True,
            p=2,
            entreg=1e-1,
            device="cpu",
            # min_labelcount=0,
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
    train, od_test = split_from_swaps(id_train, id_test, target, 1000, 20, 20)
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

    # doesn't work yet
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


def split_from_swaps(train, test, target, n_swaps=100, swaps_per_iter=20, history=10):
    """
    Function adapted from
    https://github.com/Confusezius/Characterizing_Generalization_in_DeepMetricLearning
    Performs data swapping between train and test sets based on distance metrics.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        target (str): Name of the target column.
        n_swaps (int): Number of swapping iterations. Default is 100.
        swaps_per_iter (int): Number of swaps per iteration. Default is 20.
        history (int): Number of previous swaps to consider for preventing duplicate swaps. Default is 10.

    Returns:
        pd.DataFrame: Modified training dataset after swapping.
        pd.DataFrame: Modified test dataset after swapping.
    """
    train_hist, test_hist = [], []
    feat_cols = train.columns[train.columns != target]
    metric = lambda x, y: custom_distance(x, y, feat_cols[7:])
    # metric = "euclidean"

    for i in range(n_swaps):
        trainmean = train.loc[:, feat_cols].mean().values.reshape(1, -1)
        testmean = test.loc[:, feat_cols].mean().values.reshape(1, -1)
        dists_train_trainmean = pairwise_distances(
            train.loc[:, feat_cols],
            trainmean,
            metric=metric,
        )
        dists_train_testmean = pairwise_distances(
            train.loc[:, feat_cols], testmean, metric=metric
        )
        dists_test_trainmean = pairwise_distances(
            test.loc[:, feat_cols], trainmean, metric=metric
        )
        dists_test_testmean = pairwise_distances(
            test.loc[:, feat_cols], testmean, metric=metric
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
    # test command
    # write_dataset_csv(
    #     ["--read_path", "../data", "--write_base", "../data", "--data", "0"]
    # )
    # OR
    # python dave/utils/mb_data_process.py --base_path=./dave/proxies --write_path="./data"
    split()
    # c1 = {"Al": 2, "O": 3}
    # c2 = {"Na": 1, "Cl": 1}
    # print(custom_levenshtein(c1, c2))
