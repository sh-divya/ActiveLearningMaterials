import sys
from pathlib import Path

DAVE_PATH = Path(__file__).parent.parent
sys.path.append(str(DAVE_PATH))

from dave.proxies.data import CrystalFeat
from torch.utils.data import DataLoader
import argparse
import torch

from pymatgen.core.periodic_table import Element

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--csv_path")
    args = parser.parse_args()
    csv_path = Path(args.csv_path).resolve()
    args = args.name

    targets = {
        "matbench_mp_e_form": "Eform",
        "matbench_mp_gap": "Band Gap",
        "mp20": "formation_energy_per_atom",
        "nrcc_ionic_conductivity": "Ionic conductivity (S cm-1)",
    }

    csv = csv_path / args

    data = CrystalFeat(csv, targets[args], subset="train")
    loader = DataLoader(data, batch_size=len(data))
    for x, y in loader:
        m1 = x[-2].mean(dim=0)
        s1 = x[-2].std(dim=0)
        torch.save(m1, csv / "x.mean")
        torch.save(s1, csv / "x.std")

        m2 = y.mean(dim=0)
        s2 = y.std(dim=0)
        torch.save(m2, csv / "y.mean")
        torch.save(s2, csv / "y.std")
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        print(x[3].shape)
