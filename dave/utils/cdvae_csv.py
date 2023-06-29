import pandas as pd
import os.path as osp
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure


def write_data_csv(root):
    data_path = osp.join(root, "data")
    subsets = ["train", "val", "test"]
    proxy_features = {
        "Space Group": [],
        "a": [],
        "b": [],
        "c": [],
        "alpha": [],
        "beta": [],
        "gamma": [],
    }

    cols_of_interest = [
        "material_id",
        "heat_all",
        "heat_ref",
        "formation_energy_per_atom",
        "band_gap",
        "e_above_hull",
        "energy_per_atom",
    ]
    for i in range(1, 119):
        proxy_features[Element("H").from_Z(i).symbol] = []
    master_df = []
    sub_lens = []
    for sub in subsets:
        id_cif_prop = pd.read_csv(osp.join(data_path, sub + ".csv"))
        df_cols = id_cif_prop.columns
        sub_cols = [col for col in cols_of_interest if col in df_cols]
        sub_df = id_cif_prop[sub_cols]
        for idx, row in id_cif_prop.iterrows():
            struc = Structure.from_str(row["cif"], fmt="cif")
            lattice = struc.lattice
            a, b, c = lattice.abc
            proxy_features["a"].append(a)
            proxy_features["b"].append(b)
            proxy_features["c"].append(c)
            alpha, beta, gamma = lattice.angles
            proxy_features["alpha"].append(alpha)
            proxy_features["beta"].append(beta)
            proxy_features["gamma"].append(gamma)
            sg = struc.get_space_group_info()[1]
            proxy_features["Space Group"].append(sg)
            comp = struc.composition
            for k in list(proxy_features.keys())[7:]:
                try:
                    proxy_features[k].append(comp[k])
                except KeyError:
                    proxy_features[k].append(0)
        lens = [len(val) for k, val in proxy_features.items()]
        df = pd.DataFrame.from_dict(proxy_features)
        sub_df = sub_df.astype({"material_id": "str"})
        master_df.append(sub_df)
        sub_lens.append(len(sub_df))
    master_df = pd.concat(master_df, axis=0, ignore_index=True)
    master_df = pd.concat([master_df, df.loc[:, (df != 0).any()]], axis=1)
    for i, l in enumerate(sub_lens):
        if i == 0:
            low = 0
        else:
            low = high
        high = l + low
        df = master_df.iloc[low:high, :]
        df.to_csv(osp.join(root, subsets[i] + "_data.csv"))