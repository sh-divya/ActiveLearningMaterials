from pymatgen.analysis.local_env import CrystalNN
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
import os.path as osp
import pandas as pd
import warnings
import os

API_KEY = 'ZOyRKuvwhXfX8jJPK'
data_base = '/Users/divya-sh/Documents/' \
             'MILA - AI for Materials/data/'
mpr = MPRester(API_KEY)

def download(criteria, properties, save_dir):
    write_path = osp.join(data_base, save_dir)
    for i, d in enumerate(mpr.query(criteria=criteria, properties=["material_id" , "cif"] + properties)):
        with open(osp.join(write_path, d['material_id'] + '.cif'), 'w+') as fobj:
            fobj.write(d['cif'])

def oracle_data(crystal):
    nn_calc = CrystalNN()
    li_count = 0
    llb = 0
    sbi = 0
    bond_count = 0
    lasd = 0
    llsd = 0
    afc = 0
    anion_count = 0
    for e, elem in enumerate(crystal):
        nb = crystal.get_neighbors(elem, 4)
        if elem.specie.Z == 3:
            cn = nn_calc.get_nn_info(crystal, e)
            # print(cn)
            min_dist1 = 1000
            min_dist2 = 1000
            for neigh in cn:
                ion = neigh['site'].specie.oxi_state
                dist = crystal.get_distance(e, neigh['site_index'])
                if ion < 0:
                    if dist < min_dist1:
                        min_dist1 = dist
                elif neigh['site'].specie.Z == 3:
                    if dist < min_dist2:
                        min_dist2 = dist
                # except AttributeError:
            lasd += min_dist1
            llsd += min_dist2
            # print(cn)
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

def builder_data(crystal):
    # print(crystal.formula)
    composition = {i:0 for i in range(1, 119)}
    denom = len(crystal)
    
    for elem in crystal:
        e = elem.specie.Z
        # print(type(e))
        composition[e] += 1 / denom

    # print(crystal.get_space_group_info()[0])
    composition['space_group'] = crystal.get_space_group_info()[0]
    return composition

def iter_data(data_dir):
    warnings.filterwarnings('ignore')
    read_path = osp.join(data_base, data_dir)
    for c, cif in enumerate(os.listdir(read_path)):
        struc = osp.join(read_path, cif)
        struc = Structure.from_file(struc)
        struc.add_oxidation_state_by_guess()
        # print(struc.species)
        # builder_data(struc)
        oracle_data(struc)
        if c == 100:
            break


if __name__ == '__main__':
    destination = 'li-ssb'

    query_crit = {
        "elements":{
            "$all":['Li']
            }
    }

    # extra_properties = []
    # download(query_crit, extra_properties, destination)

    iter_data(destination)