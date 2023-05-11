import os
import os.path as osp
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_splits = ["train_data.csv", "val_data.csv", "test_data.csv"]
    data_dir = {
        "carbon": "energy_per_atom",
        "matbench_mp_e_form": "Eform",
        "matbench_mp_gap": "Band Gap",
        "mp20": "formation_energy_per_atom",
        "perov": "heat_all",
    }

    feats = ["Space Group", "a", "b", "c", "alpha", "beta", "gamma"]
    for d, t in data_dir.items():
        dir_path = Path(osp.join("./proxies", d))
        params = [t] + feats
        fig, ax = plt.subplots(2, 4, figsize=(15, 6))
        for i, p in enumerate(params):
            curr_ax = ax[i // 4][i % 4]
            for split in data_splits:
                csv_path = dir_path / split
                df = pd.read_csv(csv_path)[p]
                # df = df.loc[df != 0]
                curr_ax.hist(
                    df,
                    bins=200,
                    histtype="stepfilled",
                    label=split.split(".")[0].split("_")[0],
                    alpha=0.5,
                    density=True,
                )
            curr_ax.set_xlabel(p)
        fig.suptitle(d)
        lines, labels = curr_ax.get_legend_handles_labels()
        fig.legend(lines, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(Path("../plots") / (f"{d}.png"))
        plt.close()
