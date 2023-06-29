import json
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import lmdb
from pymatgen.core.structure import Structure
from tqdm import tqdm
from matbench.metadata import MBV01_VALIDATION_DATA_PATH

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dave.utils.misc import resolve

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--output_lmdb", type=str)
    args = parser.parse_args()

    json_path = resolve(args.input_json)
    assert json_path.exists()

    db_path = resolve(args.output_lmdb)
    assert not db_path.exists(), f"{str(db_path)} already exists!"
    if not db_path.parent.exists():
        print(f"Creating parent directory for {str(db_path)}")
        db_path.parent.mkdir(parents=True)

    print(f"Reading data from {str(json_path)}...", end=" ", flush=True)
    data = json.loads(json_path.read_text())
    print("Done.")

    print(f"Writing data to {str(db_path)}:", flush=True)
    db = lmdb.open(
        str(db_path),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)

    for i, (x, y) in tqdm(zip(data["index"], data["data"]), total=len(data["index"])):
        s = Structure.from_dict(x)
        txn.put(
            f"{i}".encode("ascii"),
            pickle.dumps((s, y), protocol=-1),
        )

    txn.commit()
    db.sync()
    db.close()
