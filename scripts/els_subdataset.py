import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

try:
    from rich.console import Console
except ImportError:
    print("rich not installed, install with `pip install rich`")
    sys.exit(1)


console = Console()
baseprint = print
print = console.log


def now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--input_csv", type=str, required=True)
    args.add_argument("--elements", type=str, default="O,H,F,S,Li,P,Mg,C,N,Cl,Si,Fe")
    args = args.parse_args()

    input_path = Path(args.input_csv)
    assert input_path.exists(), f"--input_csv must exist ({input_path})"
    assert input_path.is_file(), f"--input_csv must be a file ({input_path})"
    assert input_path.suffix == ".csv", f"--input_csv must be a csv ({input_path})"

    elements = args.elements.split(",")
    assert len(elements) > 0, "Must specify at least one element"

    output_path = (
        Path(args.input_csv).parent
        / "sub-datasets"
        / f"{input_path.stem}-{len(elements)}-{'_'.join(elements)}.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Output path {output_path} already exists", style="bold red")
        if not console.input("Overwrite? [y/n] ").lower().startswith("y"):
            sys.exit(1)

    print(f"Reading csv from {input_path} with pandas", style="green")
    with console.status("Loading"):
        df = pd.read_csv(input_path)
        n_total = len(df)

    cols = df.columns.to_list()
    edf = df[df.columns[cols.index("H") : cols.index("Eform")]]

    print(f"Total number of rows: {len(edf)}")
    print(f"Number of referenced elements: {len(edf.columns)}")
    print(f"Number of non-zero elements: {np.count_nonzero(edf.sum(0))}")

    assert all(
        e in edf.columns for e in elements
    ), "Not all elements are in the dataset"

    print(f"Filtering for elements: {elements}")
    els_set = set(elements)
    remaing_elements = [c for c in edf.columns if c not in els_set]
    df = df[df[remaing_elements].sum(1) == 0]
    print(f"Remaining number of rows: {len(df)} ({len(df) / n_total * 100:.2f}%)")

    df.to_csv(output_path, index_label="original_index")
    print(f"Saved to {output_path}", style="bold green")

    config_path = output_path.parent / f"args/{now()}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        from yaml import safe_dump

        safe_dump(vars(args), f)

    print(f"Config saved to {config_path}", style="italic yellow")
