import json
from pathlib import Path

from cdvae_csv import feature_per_struc, FEATURE_KEYS

import click
import pandas as pd
from pymatgen.core.structure import Structure


@click.command()
@click.option("--read_path", default="./data")
@click.option("--write_base", default="./proxies")
def write_dataset_csv(read_path, write_base):
    db_path = Path(read_path)
    write_base = Path(write_base)
    db_name = ["matbench_mp_e_form", "matbench_mp_gap"]
    targets = ["Eform", "Band Gap"]

    for i, db in enumerate(db_name):
        y = []
        proxy_features = {key: [] for key in FEATURE_KEYS}
        json_file = db_path / (db + ".json")
        with open(json_file, "r") as fobj:
            jobj = json.load(fobj)
            for j in jobj["data"][:35]:
                struc = Structure.from_dict(j[0])
                proxy_features = feature_per_struc(struc, proxy_features)
                y.append(j[1])
        proxy_features[targets[i]] = y
        df = pd.DataFrame.from_dict(proxy_features)
        df = df.loc[:, (df != 0).any()]
        df.to_csv(write_base / db / "data" / (db + ".csv"))


if __name__ == "__main__":
    write_dataset_csv()
