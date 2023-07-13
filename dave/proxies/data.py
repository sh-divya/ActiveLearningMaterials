import os
import sys
import gzip
import torch
import tempfile
import pandas as pd
import os.path as osp
from pathlib import Path
from urllib import request
from typing import Callable, List, Any, Sequence
from pymatgen.core.structure import Structure

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset

DAVE_PATH = Path(__file__).parent.parent
sys.path.append(str(DAVE_PATH / "utils"))

from atoms_to_graph import AtomsToGraphs, pymatgen_structure_to_graph
from mb_data_process import write_dataset_csv, split


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


class CrystalGraph(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        name="mp20",
        subset="train",
    ):
        self.name = name
        self.subset = subset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        check_path = Path(self.root) / "data" / f"{self.name}"
        return str(check_path)

    @property
    def processed_dir(self) -> str:
        check_path = Path(self.root) / "dave" / "proxies" / f"{self.name}"
        return str(check_path)

    @property
    def raw_file_names(self):
        if self.name == "mp20":
            return [f"{self.subset}.csv"]
        else:
            return [f"{self.subset}_data.csv"]

    @property
    def processed_file_names(self):
        return [self.subset + ".pt"]

    def download(self):
        temp_raw = Path(self.raw_dir)
        temp_raw.mkdir(parents=True, exist_ok=True)
        if self.name == "mp20":
            download_url(
                f"https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/{self.subset}.csv",
                self.raw_dir,
            )
        if self.name == "matbench_mp_e_form":
            print(self.raw_dir)
            print(self.raw_paths)
            request.urlretrieve(
                "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz",
                temp_raw / "matbench_mp_e_form.json.gz",
            )
            # download_url(
            #     "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz",
            #     self.raw_dir,
            # )
            fd, tmp_path = tempfile.mkstemp(dir=self.raw_dir)
            with os.fdopen(fd, "w") as tmp:
                tmp.write(
                    gzip.open(
                        osp.join(self.raw_dir, "matbench_mp_e_form.json.gz"), "rb"
                    )
                )
                raw_parent = Path(self.raw_dir).parent
                write_dataset_csv(
                    read_path=self.raw_dir, write_base=str(raw_parent), data=0
                )
                split(base_path=raw_parent, data_select="0")
            os.close(fd)
            os.remove(tmp_path)

    def process(self):
        a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6.0,
            r_energy=False,
            r_forces=False,
            r_distances=True,
            r_edges=False,
        )
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))
        data_list = []
        for idx, row in data_df.iterrows():
            struct = Structure.from_str(row["cif"], fmt="cif")
            if self.name == "mp20":
                target = row["formation_energy_per_atom"]
            else:
                target = row["Eform"]
            data = pymatgen_structure_to_graph(struct, a2g)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
