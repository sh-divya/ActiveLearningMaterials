import os
import os.path as osp
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

FEATS = ["Space Group", "a", "b", "c", "alpha", "beta", "gamma"]


def plots_from_df(dataf, target, ax, split):
    params = FEATS + [target]
    for i, p in enumerate(params):
        curr_ax = ax[i // 4][i % 4]
        df = dataf[p]
        curr_ax.hist(
            df, bins=200, histtype="stepfilled", label=split, alpha=0.5, density=True
        )
        curr_ax.set_xlabel(p)

    lines, labels = curr_ax.get_legend_handles_labels()

    return lines, labels


def all_plots():
    data_splits = ["train_data.csv", "val_data.csv", "test_data.csv"]
    data_dir = {
        "carbon": "energy_per_atom",
        "matbench_mp_e_form": "Eform",
        "matbench_mp_gap": "Band Gap",
        "mp20": "formation_energy_per_atom",
        "perov": "heat_all",
    }

    for d, t in data_dir.items():
        fig, ax = plt.subplots(2, 4, figsize=(15, 6))
        dir_path = Path(osp.join("./proxies", d))
        for split in data_splits:
            csv_path = dir_path / split
            df = pd.read_csv(csv_path)
            lines, labels = plots_from_df(df, t, ax, split.split("_")[0])
        fig.legend(lines, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(Path("../plots") / (f"{d}.png"))
        plt.close()


if __name__ == "__main__":
    all_plots()
