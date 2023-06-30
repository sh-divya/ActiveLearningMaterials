from pymatgen.ext.matproj import MPRester
import os.path as osp
import click


def download(queryObj, criteria, properties, save_dir):
    write_path = osp.join(save_dir)
    for i, d in enumerate(
        queryObj.query(
            criteria=criteria, properties=["material_id", "cif"] + properties
        )
    ):
        with open(osp.join(write_path, d["material_id"] + ".cif"), "w+") as fobj:
            fobj.write(d["cif"])


@click.command()
@click.option("--filepath", default="./data")
@click.option("--apikey_filepath", default="./apikey.txt")
def data_setup(filepath, apikey_filepath):
    with open(apikey_filepath) as fobj:
        apikey = fobj.read().strip()
        mpr = MPRester(apikey)
        query_crit = {"elements": {"$all": ["Li"]}}

        extra_properties = []
        download(mpr, query_crit, extra_properties, filepath)
