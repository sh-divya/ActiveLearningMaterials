import pickle
import torch
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import lmdb


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


class MatBenchDataset(Dataset):
    def __init__(self, src, indices, scalex=False, scaley=False, n_els=103):
        self.xtransform = scalex
        self.ytransform = scaley
        self.src = src
        self.indices = indices
        self.n_els = n_els

        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = lmdb.open(
                str(self.src),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
        return self._db

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        struct, target = pickle.loads(
            self.db.begin().get(f"{self.indices[idx]}".encode("ascii")),
        )

        target = torch.tensor(target, dtype=torch.float32)

        sg = struct.get_space_group_info()[1]
        lat = torch.tensor(
            struct.lattice.lengths + struct.lattice.angles, dtype=torch.float32
        )
        str_to_comp = struct.composition.get_el_amt_dict()
        comp = torch.zeros(self.n_els, dtype=torch.int32)
        for el in struct.composition.elements:
            comp[el.number - 1] = str_to_comp[el.name]
        comp = comp.to(torch.int32)

        if self.xtransform:
            lat = ((lat - self.xtransform["mean"]) / self.xtransform["std"]).to(
                torch.float32
            )
        if self.ytransform:
            target = ((target - self.ytransform["mean"]) / self.ytransform["std"]).to(
                torch.float32
            )
        return (comp, sg, lat), target


if __name__ == "__main__":
    from dave.utils.misc import set_cpus_to_workers
    from dave.utils.loaders import make_loaders

    config = {
        "config": "physmlp-matbench",
        "scales": {
            "x": False,
            "y": False,
        },
        "src": "/Users/victor/Documents/Github/ActiveLearningMaterials/data/matbench_mp_e_form.lmdb",
        "val_frac": 0.20,
        "fold": 0,
        "optim": {"batch_size": 32},
    }
    config = set_cpus_to_workers(config)
    loaders = make_loaders(config)
