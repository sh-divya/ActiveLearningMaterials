import os.path as osp

import numpy as np
import pandas as pd
import torch
from mendeleev.fetch import fetch_table
from torch.utils.data import Dataset
import re

def parse_sample(data, target):
    parsed_data = []
    elem_df = fetch_table("elements")	
    all_elems = elem_df['symbol']

    pat = re.compile("|".join(all_elems.tolist()))
    
    for s, sample in data.iterrows():
        try:
            comp = sample["Formulae"]
            print(comp)
        except KeyError:
            comp = sample["Composition"]
        match = re.findall(pat, comp)
        stoich = re.split(pat, comp)[1:]
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
        
        for e in all_elems:
            dix[e] = 0
        for e, f in zip(match, stoich):
            tmp = re.match(r"([a-z]+)([0-9]+)", f, re.I)
            if tmp:
                items = tmp.groups()
                dix[e + items[0]] = float(items[1])
            else:
                dix[e] = float(f)
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
            "DOI"
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
        if self.xtransform:
            lat = ((lat - self.xtransform["mean"]) / self.xtransform["std"]).to(
                torch.float32
            )
        if self.ytransform:
            target = ((target - self.ytransform["mean"]) / self.ytransform["std"]).to(
                torch.float32
            )
        return (comp, sg, lat), target