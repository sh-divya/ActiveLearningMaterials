import pandas as pd
from pathlib import Path
import argparse

COMP_START = 7


def dave_to_standard(data, keep_target=False, keep_cif=False):
    columns = data.columns
    comp_df = data[columns[COMP_START:-2]]
    formulae = comp_df.apply(lambda x: convert_comp(x, columns[COMP_START:-2]), axis=1)
    drop_cols = list(map(str, columns[COMP_START:-2]))
    suffix = ""
    if not keep_target:
        drop_cols.append(str(columns[-2]))
    else:
        suffix += f"_{str(columns[-2])}"
    if not keep_cif:
        drop_cols.append(str(columns[-1]))
    else:
        suffix += f"_{str(columns[-1])}"
    data = data.drop(labels=drop_cols, axis=1)
    data.insert(0, "Formulae", formulae)
    return data, suffix


def convert_comp(comp, elements):
    return "".join([f"{elements[k]}{str(int(i))}" for k, i in enumerate(comp) if i > 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dave/proxies/matbench_mp_eform")
    parser.add_argument("--csv", default="train_data.csv")
    parser.add_argument("--cif", action="store_true")
    parser.add_argument("--target", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    csv = args.csv
    df = pd.read_csv(str(root / csv), index_col=0)
    std_data, suff = dave_to_standard(df, keep_target=args.target, keep_cif=args.cif)
    main = csv.split("_")[0]
    new_name = str(root / f"std_{main}{suff}.csv")
    std_data.to_csv(new_name)
