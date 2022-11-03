from pymatgen.ext.matproj import MPRester
from torch.utils.data import Dataset
from pymatgen.core import Structure
import matplotlib.pyplot as plt
import os.path as osp
import torch.nn as nn
import numpy as np
import warnings
import torch
import click
import os


class CrystalDataset(Dataset):
    def __init__(self, data_path='./data', csv_path='./csvfile.csv', skip=False, skip_fptr='skip.txt'):
        self.data_path = data_path
        self.cifs = os.listdir(data_path)
        self.write_csv = csv_path
        self.oracleSendek = nn.Sequential(
            nn.Linear(5, 1, bias=True),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.oracleSendek[0].weight = nn.Parameter(torch.tensor([0.18, -4.01, -0.47, 8.70, -2.17]).float())
            self.oracleSendek[0].bias = nn.Parameter(torch.tensor(-6.56).float())
        self.skip = skip
        if not skip:
            if osp.isfile(skip_fptr):
                self.skipObj = open(skip_fptr, 'a')
            else:
                self.skipObj = open(skip_fptr, 'w+')
        else:
            self.skipObj = open(skip_fptr, 'r')

    def oracle_data(self, crystal):
        llb = []
        sbi = []
        lasd = []
        llsd = []
        afc = []
        anions = [(x.specie.Z, x.specie.X) for x in crystal if x.specie.X >= 0]
        A = max(anions, key=lambda x: x[1])[0]
        crystal_neighs = crystal.get_all_neighbors(12)

        for s, site in enumerate(crystal_neighs):
            if crystal[s].specie.Z == 3:
                llsd.append(min([neigh[1] for neigh in site if neigh.specie.Z == 3]))
                lasd.append(min([neigh[1] for neigh in site if neigh.specie.Z == A]))

        for e, elem in enumerate(crystal):
            
            if elem.specie.Z == 3:
                nb = crystal.get_neighbors(elem, 4)
                llb.append(sum([i.specie.Z  == 3 for i in nb]))
            else:
                nb = crystal.get_neighbors(elem, 4)
                sbi.extend([abs(elem.specie.X - i.specie.X) for i in nb])
            if elem.specie.Z == A:
                rij0 = min([neigh[1] for neigh in crystal_neighs[e] if neigh.specie.Z == A])
                nb = crystal.get_neighbors_in_shell(elem.coords, rij0, 1)
                afc.append(sum([i.specie.Z == A for i in nb]))

        llb = np.mean(llb)
        lasd = np.mean(lasd)
        llsd = np.mean(llsd)
        sbi = np.mean(sbi)
        afc = np.mean(afc)

        return (llb, sbi, afc, lasd, llsd)

    def building_blocks(self, crystal):
        # print(crystal.formula)
        composition = {i:0 for i in range(1, 119)}
        denom = len(crystal)
        
        for elem in crystal:
            e = elem.specie.Z
        composition[e] += 1 / denom

        composition['space_group'] = crystal.get_space_group_info()[1]
        return composition

    def populate(self):
        warnings.filterwarnings('ignore')
        fobj = open(self.write_csv, 'a')
        fobj.write('\t'.join(['ID', 'LLB', 'SBI', 'AFC', 'LASD', 'LLSD', 'Psuperionic']) + '\n')
        if self.skip:
            skip_cifs  = [line.split()[0] for line in self.skipObj]
        else:
            skip_cifs = []
        for cif in self.cifs[1000:]:
            if self.skip:
                if cif in skip_cifs:
                    continue
            struc = osp.join(self.data_path, cif)
            struc = Structure.from_file(struc)
            struc.add_oxidation_state_by_guess()
            try:
                oracle = self.oracle_data(struc)
                y = self.oracleSendek(torch.tensor(oracle).float())
                line = '\t'.join([cif.split('.')[0]] + list(map(str, oracle))) + '\t' + str(y.data.item()) + '\n'
                fobj.write(line)
            except:
                self.skipObj.write(cif.split('.')[0] + '\tCalculation\n')
                continue
        fobj.close()

    def verify_structs(self):
        for cif in self.cifs:
            # print(cif)
            warnings.filterwarnings('error')
            struc = osp.join(self.data_path, cif)
            try:
                struc = Structure.from_file(struc)
                check_sites = sum([not site.is_ordered for site in struc])
                if check_sites > 0:
                    self.skipObj.write(cif.split()[0] + '\tDisordered\n')
            except UserWarning:
                self.skipObj.write(cif.split()[0] + '\tReadWarning\n')
                continue


def download(queryObj, criteria, properties, save_dir):
    write_path = osp.join(save_dir)
    for i, d in enumerate(queryObj.query(criteria=criteria, properties=["material_id" , "cif"] + properties)):
        with open(osp.join(write_path, d['material_id'] + '.cif'), 'w+') as fobj:
            fobj.write(d['cif'])


@click.command()
@click.option('--filepath', default='./data')
@click.option('--apikey_filepath', default='./apikey.txt')
def data_setup(filepath, apikey_filepath):
    with open(apikey_filepath) as fobj:
        apikey = fobj.read().strip()
        mpr = MPRester(apikey)
        query_crit = {
            "elements":{
                "$all":['Li']
                }
        }

        extra_properties = []
        download(mpr, query_crit, extra_properties, filepath)


@click.command()
@click.option('--datapath', default='./data')
@click.option('--csvfile', default='./temp.csv')
@click.option('--avoidfile', default='./skip.txt')
@click.option('--avoid', default=False)
def process_data(datapath, csvfile, avoidfile, avoid):
    dataObj = CrystalDataset(datapath, csvfile, avoid, avoidfile)
    # dataObj.verify_structs()
    # dataObj.skipObj.close()
    dataObj.populate()


def verify_sendek():
    sendek_candidates = {
        'mp-554076': 0.589,
        'mp-532413': 0.897,
        'mp-569782': 1, #a
        'mp-558219': 0.518,
        'mp-15797': 0.543,
        'mp-29410': 0.994,
        'mp-676361': 0.655,
        'mp-643069': 0.652, #a
        'mp-19896': 0.604,
        'mp-7744': 1, #a
        'mp-22905': 0.837, #b
        'mp-34477': 0.89,
        'mp-676109': 0.656,
        'mp-559238': 0.812,
        'mp-866665': 1, # a
        'mp-8751': 0.775,
        'mp-15789': 0.901,
        'mp-15790': 0.899,
        'mp-15791': 0.899,
        'mp-561095': 0.984 , # a
        'mp-8430': 0.76
    }

    temp = CrystalDataset('./data/li-ssb', 'lissb.csv', skip=False, skip_fptr='skip.txt')
    y = []
    for mp in sendek_candidates:
        warnings.filterwarnings('ignore')
        path = osp.join('./data/li-ssb', mp + '.cif')
        struc = Structure.from_file(path)
        struc.add_oxidation_state_by_guess()
        oracle = temp.oracle_data(struc)
        y.append(temp.oracleSendek(torch.tensor(oracle).float()).data.item())

    y_sendek = list(sendek_candidates.values())
    error = [i - j for i, j in zip(y, y_sendek)]
    plt.scatter(y_sendek, y, label='Sendek vs Divya', color='k')
    # plt.scatter()
    plt.plot(y_sendek, y_sendek, label='No Error')
    error = [abs(e) for e in error]
    minE = min(error)
    maxE = max(error)
    meanE = np.mean(error)
    plt.plot([], [], ' ' , label=f'Min Error {minE:.3f}')
    plt.plot([], [], ' ' , label=f'Max Error {maxE:.3f}')
    plt.plot([], [], ' ' , label=f'MAE: {meanE:.2f}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # data_setup()
    # process_data()
    verify_sendek()
