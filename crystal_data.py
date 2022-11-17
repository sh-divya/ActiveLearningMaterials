from pymatgen.core.periodic_table import Element
from pymatgen.ext.matproj import MPRester
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import os.path as osp
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import torch
import click
import os


class CrystalDataset(Dataset):
    def __init__(self, data_path='./data', true_path='./csvfile.csv',
                 skip=False, skip_fptr='skip.txt', feat_path='./proxy.csv',
                 final_path='./compile.csv', transform=None, subset=False):
        self.data_path = data_path
        self.cifs = os.listdir(data_path)
        self.true_csv = true_path
        self.feat_csv = feat_path
        self.compile_csv = final_path
        self.oracleSendek = nn.Sequential(
            nn.Linear(5, 1, bias=True),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.oracleSendek[0].weight = nn.Parameter(torch.tensor([0.18, -4.01, -0.47, 8.70, -2.17]).float())
            self.oracleSendek[0].bias = nn.Parameter(torch.tensor(-6.56).float())
        self.skip = skip
        self.transform = transform
        if not skip:
            if osp.isfile(skip_fptr):
                self.skipObj = open(skip_fptr, 'a')
            else:
                self.skipObj = open(skip_fptr, 'w+')
        else:
            self.skipObj = open(skip_fptr, 'r')
            self.skip_crys = [i.split()[0] for i in self.skipObj]
            self.mat_df = pd.read_csv(self.compile_csv)
            self.mat_df = self.mat_df.loc[:, (self.mat_df != 0).any(axis=0)]
            self.mat_df = self.mat_df.dropna(axis=0, subset=['Psuperionic'])
            cols = self.mat_df.columns[self.mat_df.columns != 'ID']
            self.mat_df[cols] = self.mat_df[cols].astype('float64')
            self.subset = subset

    def __len__(self):
        return len(self.mat_df.index)

    def get_all_y(self):
        return self.mat_df.iloc[:, -1] >= 0.5


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

    def building_blocks(self):
        proxy_features = {'ID': [], 'Li content': [], 'Natoms': [], 'Space Group': [],
                        'a': [], 'b': [], 'c': [], 'alpha': [], 'beta': [],
                        'gamma':[]}
        for i in range(1, 119):
            if i != 3:
                proxy_features[Element('H').from_Z(i).symbol] = []
        for m, mp in enumerate(self.cifs[:]):
            warnings.filterwarnings('ignore')
            path = osp.join(self.data_path, mp)
            struc = Structure.from_file(path)
            lattice = struc.lattice
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            proxy_features['a'].append(a) 
            proxy_features['b'].append(b) 
            proxy_features['c'].append(c) 
            proxy_features['alpha'].append(alpha) 
            proxy_features['beta'].append(beta)
            proxy_features['gamma'].append(gamma) 
            proxy_features['ID'].append(mp.split('.')[0])
            proxy_features['Space Group'].append(struc.get_space_group_info()[1])
            comp = struc.composition
            proxy_features['Natoms'].append(comp.num_atoms)
            comp_dix = dict(comp.fractional_composition.as_dict())
            proxy_features['Li content'].append(comp_dix['Li'])
            for elem in list(proxy_features.keys())[10:]:
                if elem in comp_dix:
                    proxy_features[elem].append(comp_dix[elem])
                else:
                    proxy_features[elem].append(0)
        df = pd.DataFrame.from_dict(proxy_features)
        df.to_csv(self.feat_csv, index=False)


    def populate(self):
        warnings.filterwarnings('ignore')
        fobj = open(self.true_csv, 'a')
        fobj.write('\t'.join(['ID', 'LLB', 'SBI', 'AFC', 'LASD', 'LLSD', 'Psuperionic']) + '\n')
        skip_cifs = []
        for cif in self.cifs[1000:]:
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

    def compile(self):
        df_true = pd.read_csv(self.true_csv, sep=r'\s+', engine='python', header=0)
        df_feat = pd.read_csv(self.feat_csv)
        df = pd.merge(df_feat, df_true[['ID', 'Psuperionic']], on='ID', how='right')
        df.to_csv(self.compile_csv, index=False)

    def __getitem__(self, idx):
        mat = self.mat_df.iloc[[idx], 1:]
        # mat = pd.to_numeric(mat)
        # print(mat)
        # print(type(mat))
        mat, y = mat.iloc[:, :-1].values, mat.iloc[:, -1].values
        # print('Here')

        if self.subset:
            # print(mat.shape)
            mat = mat[:, [0, 1] + list(range(10, 93))]

        if self.subset:
            mat = torch.from_numpy(mat).unsqueeze(0)
        else:
            mat = torch.tensor(mat).squeeze(0)
            mat = torch.cat((
                mat[:2],
                one_hot((mat[2] - 1).to(torch.int64), 230),
                mat[3:]
            ))
        if self.transform:
            mat = (mat - self.transform['mean']) / self.transform['std']

        return torch.nan_to_num(mat, nan=0.0).squeeze(0).squeeze(0), torch.tensor(y)



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
@click.option('--datapath', default='./data/li-ssb')
@click.option('--truecsv', default='./data/lissb.csv')
@click.option('--featcsv', default='./data/proxy.csv')
@click.option('--finalcsv', default='./data/compile.csv')
@click.option('--avoidfile', default='./data/skip.txt')
@click.option('--avoid', default=True)
def process_data(datapath, truecsv, avoid, avoidfile, featcsv, finalcsv):
    dataObj = CrystalDataset(datapath, truecsv, avoid, avoidfile, featcsv, finalcsv, None, True)
    # print(list(range(len(dataObj))))
    y = dataObj.get_all_y()
    print(y)
    # print(dataObj[0])
    # dataObj.verify_structs()
    # dataObj.populate()
    # dataObj.building_blocks(
    # dataObj.compile()
    # temploader = DataLoader(dataObj, batch_size=15962)
    # for x, y in temploader:
    #     m = x.mean(dim=0)
    #     s = x.std(dim=0)
    #     torch.save(m, './data/mean85.pt')
    #     torch.save(s, './data/std85.pt')


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
    process_data()
    # verify_sendek()
