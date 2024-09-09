import sys
from pathlib import Path

DAVE_PATH = Path(__file__).parent.parent
sys.path.append(str(DAVE_PATH))

from dave.proxies.data import CrystalFeat, CrystalMOD
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
        "mod_feat": "Ionic conductivity (S cm-1)"
    }

    csv = csv_path / args

    if args == "mod_feat":
        data = CrystalMOD(csv, targets[args], subset="train")
    else:
        data = CrystalFeat(csv, targets[args], subset="train")
    loader = DataLoader(data, batch_size=len(data))
    for x, y in loader:
        if args != "mod_feat":
            m1 = x[-1].mean(dim=0)
            s1 = x[-1].std(dim=0)
        else:
            a, b, c = x
            b = b.unsqueeze(-1)
            ma = a.mean(dim=0)
            mb = torch.zeros(1)
            mc = c.mean(dim=0)
            m1 = torch.cat((ma, mb, mc), dim=-1)
            sa = a.std(dim=0)
            sb = torch.ones(1)
            sc = c.std(dim=0)
            s1 = torch.cat((sa, sb, sc), dim=-1)
        
        m2 = y.mean(dim=0)
        s2 = y.std(dim=0)
        
        torch.save(m1, csv / "x.mean")
        torch.save(s1, csv / "x.std")
        torch.save(m2, csv / "y.mean")
        torch.save(s2, csv / "y.std")
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
