from pymatgen.analysis.local_env import CrystalNN
from pymatgen.ext.matproj import MPRester
from torch.utils.data import Dataset
from pymatgen.core import Structure
import os.path as osp
import pandas as pd
import warnings
import torch
import os

API_KEY = 'ZOyRKuvwhXfX8jJPK'
mpr = MPRester(API_KEY)

class CrystalDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.cifs = os.listdir(data_path)

    def oracle_data(self, crystal):
        nn_calc = CrystalNN()
        li_count = 0
        llb = 0
        sbi = 0
        bond_count = 0
        lasd = 0
        llsd = 0
        afc = 0
        anion_count = 0
        # raise Exception
        for s, site in enumerate(crystal.get_all_neighbors(8)):
            if crystal[s].specie.Z == 3:
                llsd += min([neigh[1] for neigh in site if crystal[s].specie.Z == 3])
                lasd += min([neigh[1] for neigh in site if neigh.specie.oxi_state <= 0])

        for e, elem in enumerate(crystal):
            nb = crystal.get_neighbors(elem, 4)
            if elem.specie.Z == 3:
                li_count += 1
                llb += sum([i.specie.Z  == 3 for i in nb])
            else:
                bond_count += len(nb)
                sbi += sum([abs(elem.specie.X - i.specie.X) for i in nb])
            if elem.specie.oxi_state <= 0:
                anion_count += 1
                afc += nn_calc.get_cn(crystal, e)

        llb = llb / li_count
        lasd = lasd / li_count
        llsd = llsd / li_count
        sbi = sbi / bond_count
        afc = afc / anion_count

        return (llb, sbi, afc, lasd, llsd)
        # return (lasd, llsd)

    def building_blocks(self, crystal):
        # print(crystal.formula)
        composition = {i:0 for i in range(1, 119)}
        denom = len(crystal)
        
        for elem in crystal:
            e = elem.specie.Z
            # print(type(e))
        composition[e] += 1 / denom

        # print(crystal.get_space_group_info()[0])
        composition['space_group'] = crystal.get_space_group_info()[1]
        return composition

    def __getitem__(self, idx):
        warnings.filterwarnings('ignore')
        struc = osp.join(self.data_path, self.cifs[idx])
        struc = Structure.from_file(struc)
        struc.add_oxidation_state_by_guess()
        oracle = self.oracle_data(struc)
        bb = self.building_blocks(struc)

        return self.cifs[idx], torch.tensor(oracle) , torch.tensor(list(bb.values()))


def download(criteria, properties, save_dir):
    write_path = osp.join(save_dir)
    for i, d in enumerate(mpr.query(criteria=criteria, properties=["material_id" , "cif"] + properties)):
        with open(osp.join(write_path, d['material_id'] + '.cif'), 'w+') as fobj:
            fobj.write(d['cif'])


if __name__ == '__main__':
    data_base = '/Users/divya-sh/Documents/' \
                'MILA - AI for Materials/data/'

    destination = 'li-ssb'


    query_crit = {
        "elements":{
            "$all":['Li']
            }
    }

    # extra_properties = []
    # download(query_crit, extra_properties, osp.join(data_base, destination))

    dataObj = CrystalDataset(osp.join(data_base, destination))
    for d in dataObj:
        print(d[0])