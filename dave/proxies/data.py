import os.path as osp
import re

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from mendeleev.fetch import fetch_table
from torch.utils.data import Dataset


def parse_sample(data, target):
    parsed_data = []
    elem_df = fetch_table("elements")
    all_elems = elem_df["symbol"]

    # pat = re.compile("|".join(all_elems.tolist()))

    for s, sample in data.iterrows():
        try:
            comp = sample["Formulae"]
        except KeyError:
            comp = sample["Composition"]
        dix = {}
        try:
            dix["Space Group"] = sample["Space Group"]
        except KeyError:
            dix["Space Group"] = sample["Space group number"]
        dix["a"] = sample["a"]
        dix["b"] = sample["b"]
        dix["c"] = sample["c"]
        dix["alpha"] = sample["alpha"]
        dix["beta"] = sample["beta"]
        dix["gamma"] = sample["gamma"]
        try:
            dix["Wyckoff"] = sample["Wyckoff"]
        except KeyError:
            pass
        for e in all_elems:
            dix[e] = 0
        comp = Composition(comp).get_el_amt_dict()
        for k, v in comp.items():
            dix[k] = v
        parsed_data.append(dix)
    return pd.DataFrame(parsed_data)


def composition_df_to_z_tensor(comp_df, max_z=-1):
    """
    Transforms a dataframe with missing species to a complete tensor with composition
    for all elements up to max_z.

    Args:
        comp_df (pd.DataFrame): The csv data as a DataFrame
        max_z (int, optional): Maximum atomic number in the data set. Defaults to -1.
    """
    table = fetch_table("elements").loc[:, ["atomic_number", "symbol"]]
    table = table.set_index("symbol")
    if max_z == -1:
        max_z = table.loc[comp_df.columns[-1], "atomic_number"]
    z = np.zeros((len(comp_df), max_z + 1))
    for col in comp_df.columns:
        z[:, table.loc[col, "atomic_number"]] = comp_df[col].values
    return torch.tensor(z, dtype=torch.int32)


def parse_wyckoff(wyckoff):
    table = fetch_table("elements").loc[:, ["atomic_number", "symbol"]]
    table = table.set_index("symbol")
    new_wyck = []
    wyckoff = wyckoff.split("-")
    for item in wyckoff:
        item = item.strip("()").split(",")
        z = table.at[item[0], "atomic_number"]
        w = int(item[1])
        new_wyck.append([z, w])

    for _ in range(1278 - len(wyckoff)):
        new_wyck.append([0, 0])
    return new_wyck


class CrystalFeat(Dataset):
    def __init__(
        self, root, target, write=False, subset="train", scalex=False, scaley=False
    ):
        csv_path = root
        self.subsets = {}
        self.cols_of_interest = [
            "material_id",
            "heat_all",
            "heat_ref",
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "energy_per_atom",
            "Eform",
            "Band Gap",
            "Ionic conductivity (S cm-1)",
            "cif",
            "DOI",
            "Wyckoff",
        ]
        self.root = root
        self.xtransform = scalex
        self.ytransform = scaley
        data_df = pd.read_csv(osp.join(csv_path, subset + "_data.csv"))
        self.y = torch.tensor(data_df[target].values, dtype=torch.float32)
        data_df = parse_sample(data_df, target)
        sub_cols = [
            col for col in data_df.columns if col not in set(self.cols_of_interest)
        ]
        H_index = sub_cols.index("H")  # should be 8
        # N
        self.sg = torch.tensor(data_df["Space Group"].values, dtype=torch.int32)
        # N x 6
        self.lattice = torch.tensor(
            data_df[["a", "b", "c", "alpha", "beta", "gamma"]].values,
            dtype=torch.float32,
        )
        try:
            self.wyckoff = data_df["Wyckoff"]
        except KeyError:
            self.wyckoff = None
        # N x (max_z + 1) -> H is index 1
        self.composition = composition_df_to_z_tensor(data_df[sub_cols[H_index:]])

        # To directly handle missing atomic numbers
        # missing_atoms = torch.zeros(x.shape[0], 5)
        # self.composition = torch.cat((x[:, 8:92].to(torch.int32), missing_atoms, x[:, 92:]), dim=-1)

    def __len__(self):
        return self.sg.shape[0]

    def __getitem__(self, idx):
        sg = self.sg[idx]
        lat = self.lattice[idx]
        comp = self.composition[idx]
        target = self.y[idx]
        if self.wyckoff is not None:
            wyck = torch.tensor(
                parse_wyckoff(self.wyckoff.iloc[idx]), dtype=torch.int32
            )
        else:
            wyck = None
        if self.xtransform:
            lat = ((lat - self.xtransform["mean"]) / self.xtransform["std"]).to(
                torch.float32
            )
        if self.ytransform:
            target = ((target - self.ytransform["mean"]) / self.ytransform["std"]).to(
                torch.float32
            )
        return (comp, sg, lat, wyck), target
