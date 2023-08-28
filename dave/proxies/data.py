import os
import sys
import gzip
import torch
import requests
import tempfile
import pandas as pd
import os.path as osp
from pathlib import Path
from typing import Callable, List, Any, Sequence
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader as GraphLoader


from dave.utils.atoms_to_graph import (
    AtomsToGraphs,
    pymatgen_structure_to_graph,
    collate,
    compute_neighbors,
)


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
            "cif",
        ]
        self.xtransform = scalex
        self.ytransform = scaley
        data_df = pd.read_csv(osp.join(csv_path, subset + "_data.csv"))
        self.y = torch.tensor(data_df[target].values, dtype=torch.float32)
        sub_cols = [col for col in data_df.columns if col not in self.cols_of_interest]
        x = torch.tensor(data_df[sub_cols].values)
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
        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6.0,
            r_energy=False,
            r_forces=False,
            r_distances=True,
            r_edges=False,
        )
        if transform is not None:
            self.xtransform = transform["x"]
            self.ytransform = transform["y"]
            transform = None
        else:
            self.xtransform = None
            self.ytransform = None
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
        return [f"{self.subset}_data.csv"]

    @property
    def processed_file_names(self):
        return [self.subset + ".pt"]

    def download(self):
        raw_parent = Path(self.raw_dir).parent
        temp_raw = Path(self.raw_dir) / "data"
        temp_raw.mkdir(parents=True, exist_ok=True)
        if self.name == "mp20":
            from cdvae_csv import write_data_csv

            download_url(
                f"https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/train.csv",
                temp_raw,
            )
            download_url(
                f"https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/val.csv",
                temp_raw,
            )
            download_url(
                f"https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/test.csv",
                temp_raw,
            )
            write_data_csv(self.raw_dir)
        if self.name == "matbench_mp_e_form":
            from dave.utils.mb_data_process import write_dataset_csv, split

            json_file = raw_parent / "matbench_mp_e_form.json"
            if not json_file.is_file():
                with open(str(json_file), "wb") as j:
                    r = requests.get(
                        "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz"
                    )
                    j.write(gzip.decompress(r.content))
            if not (Path(self.raw_dir) / f"data/{self.name}.csv").is_file():
                try:
                    write_dataset_csv(
                        [
                            "--read_path",
                            str(raw_parent),
                            "--write_base",
                            str(raw_parent),
                            "--data",
                            "0",
                        ]
                    )
                except SystemExit as err:
                    split(["--base_path", raw_parent, "--data_select", "0"])
                    return
            else:
                split(["--base_path", raw_parent, "--data_select", "0"])

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))
        data_list = []
        proc_dir = Path(self.processed_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in data_df.iterrows():
            struct = Structure.from_str(row["cif"], fmt="cif")
            if self.name == "mp20":
                target = "formation_energy_per_atom"
                SGA = SpacegroupAnalyzer(struct)
                struct = SGA.get_conventional_standard_structure()
            else:
                target = "Eform"
            y = torch.tensor(row[target], dtype=torch.float32)
            not_comp_cols = [
                "cif",
                target,
                "Space Group",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "material_id",
                "band_gap",
                "e_above_hull",
            ]
            data = pymatgen_structure_to_graph(struct, self.a2g)
            if self.ytransform is not None:
                data.y = ((y - self.ytransform["mean"]) / self.ytransform["std"]).to(
                    torch.float32
                )
            else:
                data.y = y
            comp = []
            for col in data_df.columns[1:]:
                if col not in not_comp_cols:
                    comp.append(row[col])
            data.comp = torch.tensor(comp, dtype=torch.int32)
            lp = torch.tensor(
                [row[i] for i in ["a", "b", "c", "alpha", "beta", "gamma"]],
                dtype=torch.int32,
            )
            if self.xtransform is not None:
                data.lp = (lp - self.xtransform["mean"] / self.xtransform["std"]).to(
                    torch.float32
                )
            else:
                data.lp = lp
            data.sg = torch.tensor(row["Space Group"], dtype=torch.int32)
            data_list.append(data)
        data, slices = self.collate(data_list)
        # data, slices = collate(data_list)
        Path(self.processed_paths[0]).parent.mkdir(exist_ok=True, parents=True)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx)
        data.neighbors = compute_neighbors(data, data.edge_index)
        return data
