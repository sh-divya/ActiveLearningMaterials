import json
import multiprocessing as mp
import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
import warnings

try:
    from rich.console import Console
except ImportError:
    print("rich not installed, install with `pip install rich`")
    sys.exit(1)

console = Console()
baseprint = print
print = console.log

sys.path.append(str(Path(__file__).resolve().parent.parent))
repo_root = Path(__file__).resolve().parent.parent

from dave.utils.misc import resolve
from dave.utils.atoms_to_graph import pymatgen_structure_to_graph, make_a2g

warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")
warnings.filterwarnings(
    "ignore", message="TqdmExperimentalWarning: rich is experimental"
)

help = dedent(
    """
    Example commands:

    $ python scripts/make_lmdbs.py \\
        --json_input /network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v1/data/matbench_mp_e_form.json
    $ python scripts/make_lmdbs.py \\
        --csv_split_input=/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v0/data/matbench_mp_e_form/train_data.csv

    $ python scripts/make_lmdbs.py \\
        --json_input /network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v1/data/matbench_mp_e_form.json \\
        --output /network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v1/dave/proxies/matbench_mp_e_form/matbench_mp_e_form-with_graphs.lmdb \\
        --to_graph \\
        --num_workers=2
    """
)


def write_lmdb(data_list_bytes, output_path):
    lmdb_env = lmdb.open(
        str(output_path),
        map_size=int(1e12),
        subdir=False,
        map_async=True,
        readahead=False,
        writemap=True,
    )
    lmdb_txn = lmdb_env.begin(write=True)
    for idx, data_bytes in enumerate(
        tqdm(
            data_list_bytes,
            desc=f"Writing to LMDB ...{'/'.join(output_path.parts[-3:])}",
            leave=False,
        )
    ):
        lmdb_txn.put(key=f"{idx}".encode("ascii"), value=data_bytes)
    lmdb_txn.commit()
    lmdb_env.close()


def dict_to_list_of_bytes(args):
    warnings.filterwarnings(
        "ignore", message="TqdmExperimentalWarning: rich is experimental"
    )
    worker_id = args["worker_id"]
    data_list = args["data_list"]
    to_graph = args["to_graph"]
    a2g = args["a2g"]

    data_list_bytes = [None for _ in range(len(data_list))]
    for idx, d in enumerate(
        tqdm(data_list, desc=f"Worker {worker_id}", position=worker_id, leave=False)
    ):
        data_bytes = {"Eform": d.pop("Eform"), "metadata": d.pop("metadata")}

        if "cif" in d:
            data_bytes["pymatgen_structure"] = Structure.from_str(d["cif"], fmt="cif")
        else:
            data_bytes["pymatgen_structure"] = Structure.from_dict(d)

        if to_graph:
            data_bytes["graph"] = pymatgen_structure_to_graph(
                data_bytes["pymatgen_structure"], a2g
            )
        data_list_bytes[idx] = pickle.dumps(d)
    return data_list_bytes


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--csv_split_input", type=str)
    parser.add_argument("--output_path", type=str, default=str(repo_root / "data"))
    parser.add_argument("--json_input", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--to_graph", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    if args.help:
        baseprint(help)
        baseprint(parser.format_help())
        sys.exit(0)

    csv_path = json_path = input_path = None
    if args.csv_split_input:
        csv_path = resolve(args.csv_split_input)
        input_path = csv_path
    if args.json_input:
        json_path = resolve(args.json_input)
        input_path = json_path

    assert csv_path or json_path, "One input required"
    assert not (csv_path and json_path), "Only one input allowed"

    output_path = resolve(args.output_path)
    if not output_path.exists():
        if "." in output_path.name:
            if not output_path.parent.exists():
                print(
                    f"Creating output directory {output_path.parent}",
                    style="bold yellow",
                )
                output_path.parent.mkdir(parents=True)
        else:
            print(f"Creating output directory {output_path}", style="bold yellow")
            output_path.mkdir(parents=True, exist_ok=True)

    if output_path.is_dir():
        output_path = output_path / f"{input_path.stem}.lmdb"

    if output_path.exists():
        print(f"Output path {output_path} already exists", style="bold red")
        if not console.input("Overwrite? [y/n] ").lower().startswith("y"):
            sys.exit(1)

    a2g = None
    if args.to_graph:
        a2g = make_a2g()

    data_list = None
    if csv_path:
        assert (
            csv_path.suffix == ".csv"
        ), f"csv_split_input must be a csv file ({csv_path})"
        assert csv_path.exists(), f"csv_split_input must exist ({csv_path})"
        assert csv_path.is_file(), f"csv_split_input must be a file ({csv_path})"
        with console.status(f"Reading csv from {csv_path} with pandas"):
            df = pd.read_csv(csv_path)
            data_list = [
                {k: d[k] for k in ["cif", "Eform"]} for d in df.to_dict("records")
            ]

    elif json_path:
        assert (
            json_path.suffix == ".json"
        ), f"json_input must be a json file ({json_path})"
        assert json_path.exists(), f"json_input must exist ({json_path})"
        assert json_path.is_file(), f"json_input must be a file ({json_path})"
        with console.status(f"Reading json from {json_path}"):
            with open(json_path, "r") as f:
                data = json.load(f)
            data_list = [{**d[0], "Eform": d[1]} for d in data["data"]]
    assert data_list is not None, "data_list is None, something went wrong"
    data_list = [
        {
            **d,
            "metadata": {"idx_in_file": i, "parent": str(input_path)},
        }
        for i, d in enumerate(data_list)
    ]
    print("👍 Data loaded", style="green")

    if args.num_workers < 2:
        print("Using single process", style="yellow")
        all_data_list_bytes = [
            dict_to_list_of_bytes(
                {
                    "worker_id": 0,
                    "data_list": data_list,
                    "to_graph": args.to_graph,
                    "a2g": a2g,
                }
            )
        ]
    else:
        data_list_per_worker = np.array_split(data_list, args.num_workers)
        print(f"Using {args.num_workers} processes", style="yellow")
        with mp.Pool(args.num_workers) as pool:
            all_data_list_bytes = pool.map(
                dict_to_list_of_bytes,
                [
                    {
                        "worker_id": worker_id,
                        "data_list": data_list_per_worker[worker_id],
                        "to_graph": args.to_graph,
                        "a2g": a2g,
                    }
                    for worker_id in range(args.num_workers)
                ],
            )
    print("👍 Data converted to bytes", style="green")
    data_list_bytes = sum(all_data_list_bytes, [])
    print("👍 Data concatenated", style="green")
    write_lmdb(data_list_bytes, output_path)
    print(f"🔥 LMDB written to {output_path}", style="bold green")
