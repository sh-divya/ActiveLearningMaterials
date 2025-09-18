from pathlib import Path
import pandas as pd
import warnings
import click
import pprint

from pymatgen.core import Composition, Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CDVAE_SET = ["mp_20", "carbon_24", "perov_5"]
ORIG_COLS = {
    "mp_20": [
        "formation_energy_per_atom",
        "band_gap",
        "cif",
    ],  # , "spacegroup.number"],
    "carbon_24": ["cif", "energy_per_atom"],
    "perov_5": ["cif", "heat_all", "heat_ref", "dir_gap", "ind_gap"],
}


def feat_from_struc(original, sg=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struc = Structure.from_str(original["cif"], fmt="cif")
    SGA = SpacegroupAnalyzer(struc)
    struc = SGA.get_conventional_standard_structure()
    lattice = struc.lattice
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    if sg:
        sg = struc.get_space_group_info()[1]
    comp1 = struc.composition.get_el_amt_dict()
    # comp2 = "".join(str(struc.composition).split(" "))
    comp1_str = ""
    for k, v in comp1.items():
        comp1_str += f"{k}{str(int(v))}"
    return comp1_str, a, b, c, alpha, beta, gamma, sg


def standardize(original):
    std_data = pd.DataFrame()
    std_data.index = original.index
    std_data[
        ["Formulae", "a", "b", "c", "alpha", "beta", "gamma", "Space Group"]
    ] = original.apply(lambda x: feat_from_struc(x, True), axis=1, result_type="expand")
    # try:
    #     std_data["Space Group"] = original["spacegroup.number"]
    # except KeyError:
    #     pass
    for col in original.columns:
        if str(col) == "band_gap":
            std_data["Band Gap"] = original["band_gap"]
        elif str(col) in ["energy_per_atom", "formation_energy_per_atom"]:
            std_data["Eform"] = original[col]
        elif str(col) != "cif":
            std_data[col] = original[col]
    std_data["cif"] = original["cif"]
    return std_data


def get_stats(data):
    struc = Structure.from_str(data["cif"], fmt="cif")
    SGA = SpacegroupAnalyzer(struc)
    struc = SGA.get_conventional_standard_structure()
    comp = struc.composition.get_el_amt_dict()
    # print(comp)
    elems = [Element(i).Z for i in comp]
    diff_elems = len(comp)
    atoms = len(struc)
    per_atom_max = max(comp.values())
    per_atom_min = min(comp.values())

    return elems, diff_elems, atoms, per_atom_max, per_atom_min


def base_change(source, data_path, task):
    if source == "all":
        data_source = CDVAE_SET[:]
    else:
        data_source = [source]

    data_path = Path(data_path)
    splits = ["train", "val", "test"]
    stats = {
        "SG": [],
        "elements": [],
        "diff_elems_max": [],
        "diff_elems_min": [],
        "atom_counts_max": [],
        "atom_counts_min": [],
        "per_elem_max": [],
        "per_elem_min": [],
        "lengths_max": [],
        "angles_max": [],
        "lengths_min": [],
        "angles_min": [],
    }
    for src in data_source:
        for s in splits:
            try:
                if task == "fmt":
                    split_csv = data_path / src / f"{s}_orig.csv"
                    orig = pd.read_csv(
                        split_csv,
                        usecols=lambda x: x != "material_id",
                        index_col="Unnamed: 0",
                    )
                    orig = orig[ORIG_COLS[source]]
                    new_data = standardize(original=orig)
                    # new_data.to_csv(data_path / src / f"{s}.csv")
                elif task == "cfg":
                    split_csv = data_path / src / f"{s}.csv"
                    orig = pd.read_csv(
                        split_csv,
                        usecols=lambda x: x != "material_id",
                        index_col="Unnamed: 0",
                    )
                    # orig = orig.iloc[:10]
                    tmp = pd.DataFrame()
                    tmp[["Elems", "Diff", "Counts", "Per_max", "Per_min"]] = orig.apply(
                        get_stats, axis=1, result_type="expand"
                    )
                    stats["elements"].extend(tmp["Elems"])
                    stats["SG"].append(orig["Space Group"].unique().tolist())
                    stats["lengths_max"].append(orig["a"].max())
                    stats["lengths_max"].append(orig["b"].max())
                    stats["lengths_max"].append(orig["c"].max())
                    stats["lengths_min"].append(orig["a"].min())
                    stats["lengths_min"].append(orig["b"].min())
                    stats["lengths_min"].append(orig["c"].min())
                    stats["angles_max"].append(orig["alpha"].max())
                    stats["angles_max"].append(orig["beta"].max())
                    stats["angles_max"].append(orig["gamma"].max())
                    stats["angles_min"].append(orig["alpha"].min())
                    stats["angles_min"].append(orig["beta"].min())
                    stats["angles_min"].append(orig["gamma"].min())
                    stats["diff_elems_max"].append(tmp["Diff"].max())
                    stats["diff_elems_min"].append(tmp["Diff"].min())
                    stats["atom_counts_max"].append(tmp["Counts"].max())
                    stats["atom_counts_min"].append(tmp["Counts"].min())
                    stats["per_elem_max"].append(tmp["Per_max"].max())
                    stats["per_elem_min"].append(tmp["Per_min"].min())
                # equality = new_data["Formulae"] != new_data["Test"]
                # print(src, s)
                # print(equality.astype(int).sum())
                # new_data = new_data.drop("Test", axis=1)
                # print(new_data)
            except pd.errors.EmptyDataError:
                print("File Doesnt Have Data")
                pass
        if task == "cfg":
            stats["SG"] = list(set([e for l in stats["SG"] for e in l]))
            stats["diff_elems_max"] = max(stats["diff_elems_max"])
            stats["diff_elems_min"] = min(stats["diff_elems_min"])
            stats["atom_counts_max"] = max(stats["atom_counts_max"])
            stats["atom_counts_min"] = min(stats["atom_counts_min"])
            stats["per_elem_max"] = max(stats["per_elem_max"])
            stats["per_elem_min"] = min(stats["per_elem_min"])
            stats["lengths_max"] = max(stats["lengths_max"])
            stats["lengths_min"] = min(stats["lengths_min"])
            stats["angles_max"] = max(stats["angles_max"])
            stats["angles_min"] = min(stats["angles_min"])
            stats["elements"] = [e for l in stats["elements"] for e in l]
            stats["elements"] = list(set(stats["elements"]))
            pprint.pprint(stats)


@click.command()
@click.option("--source", default="all")
@click.option("--data_path", default="/network/projects/crystalgfn/data/")
@click.option("--task", default="fmt")
def change(source, data_path, task):
    base_change(source, data_path, task)


if __name__ == "__main__":
    change()
