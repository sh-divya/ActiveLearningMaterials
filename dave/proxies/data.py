import os
import torch

import pandas as pd
import os.path as osp
from torch.utils.data import Dataset, DataLoader

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
    print(max_z)
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
            "IC",
            "cif",
        ]
        self.xtransform = scalex
        self.ytransform = scaley
        data_df = pd.read_csv(osp.join(csv_path, subset + "_data.csv"))
        self.y = torch.tensor(data_df[target].values, dtype=torch.float32)
        sub_cols = [col for col in data_df.columns if col not in self.cols_of_interest]
        x = torch.tensor(data_df[sub_cols].values, dtype=float)
        self.sg = x[:, 1].to(torch.int32)
        self.lattice = x[:, 2:8].float()
        self.composition = x[:, 8:].to(torch.int32)
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
