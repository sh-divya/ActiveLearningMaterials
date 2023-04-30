import os
import torch
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset, DataLoader


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


class Matbench(Dataset):
    pass


if __name__ == "__main__":
    folder = "./carbon"
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
        root=folder, target="energy_per_atom", subset="train"
    )  # , scalex=xt, scaley=yt)
    bs = len(temp)
    print(temp[10][0])
    loader = DataLoader(temp, batch_size=100)
    for x, y in loader:
        # m1 = x[-1].mean(dim=0)
        # s1 = x[-1].std(dim=0)
        torch.save(m1, osp.join(folder, "x.mean"))
        torch.save(s1, osp.join(folder, "x.std"))

        # m2 = y.mean(dim=0)
        # s2 = y.std(dim=0)
        # torch.save(m2, osp.join(folder, "y.mean"))
        # torch.save(s2, osp.join(folder, "y.std"))
