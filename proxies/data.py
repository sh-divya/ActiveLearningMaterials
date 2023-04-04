import os
import torch
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset, DataLoader


class CrystalFeat(Dataset):
    def __init__(
        self, root, target, write=False, subset="train", scalex=False, scaley=False
    ):
        self.csv_path = root
        self.subsets = {}
        self.cols_of_interest = [
            "material_id",
            "heat_all",
            "heat_ref",
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "energy_per_atom",
        ]
        self.xtransform = scalex
        self.ytransform = scaley
        self.data_df = pd.read_csv(osp.join(self.csv_path, subset + "_data.csv"))
        self.y = self.data_df[target].values
        sub_cols = [
            col for col in self.data_df.columns if col not in self.cols_of_interest
        ]
        self.x = self.data_df[sub_cols].values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        mat = torch.from_numpy(self.x[idx, 1:])
        target = torch.tensor((self.y[idx]))
        if self.xtransform:
            mat = (mat - self.xtransform["mean"]) / self.xtransform["std"]
        if self.ytransform:
            target = (target - self.ytransform["mean"]) / self.ytransform["std"]
        return torch.nan_to_num(mat, nan=0.0), target


if __name__ == "__main__":
    folder = "./perov"
    # write_data_csv(folder)
    # xt = {
    #     "mean": torch.load(osp.join(folder, "x.mean")),
    #     "std": torch.load(osp.join(folder, "x.std")),
    # }
    # yt = {
    #     "mean": torch.load(osp.join(folder, "y.mean")),
    #     "std": torch.load(osp.join(folder, "y.std")),
    # }
    temp = CrystalFeat(
        root=folder, target="heat_ref", subset="train"
    )  # , scalex=xt, scaley=yt)
    bs = len(temp)
    print(temp[10][0].shape)
    loader = DataLoader(temp, batch_size=bs)
    for x, y in loader:
        m1 = x.mean(dim=0)
        s1 = x.std(dim=0)
        torch.save(m1, osp.join(folder, "x.mean"))
        torch.save(s1, osp.join(folder, "x.std"))
        m2 = y.mean(dim=0)
        s2 = y.std(dim=0)
        torch.save(m2, osp.join(folder, "y.mean"))
        torch.save(s2, osp.join(folder, "y.std"))
