import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.linear_model import LinearRegression
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import subprocess
import click

from m3gnet.models import MolecularDynamics, M3GNet, Potential
from pymatgen.core import Structure
from ase import units
from ase.io import Trajectory
from ase.md.analysis import DiffusionCoefficient
import tensorflow as tf
import warnings
import json

warnings.filterwarnings("ignore")

def download(crystals, queryObj, save_dir):
	write_path = save_dir
	for i, c in enumerate(crystals):
		crit = c
		try:
			res = queryObj.query(criteria=crit, properties=["material_id", 'structure'])[0]
			struc = res['structure']
			struc.add_oxidation_state_by_guess()
			writeObj = CifWriter(struc)
			writeObj.write_file(osp.join(write_path, res['material_id'] + '.cif'))
		except IndexError:
			print(c)
		

@click.command()
@click.option('--dirpath', default='./data')
@click.option('--keypath', default='./apikey.txt')
def data(dirpath, keypath):
	# crystals = [
	# 	"LiI",
	# 	"Li2NH",
	# 	"Li2S",
	# 	"Li10GeP2S12",
	# 	"Li3Fe2P3O12"
	# ]
	# crystals = [
	# 	"Li5PS4Cl2",
	# 	"Li6PS5Cl",
	# 	"Li3PS4",
	# 	# "LiYCl6",
	# 	"Li10GeP2S12",
	# 	"Li7La3Zr2O12",
	# 	"Li3ScCl6"
	# 	# "Li3HoCl6",
	# 	# "Li3ScBr6",
	# 	# "Li3HoBr6",
	# ]
	crystals = [
		'mp-6674',
		'mp-8609',
		'mp-28567',
		'mp-12160',
		'mp-6787',
		'mp-560072',
		'mp-7941',
		'mp-13772',
		'mp-8452',
		'mp-770932',
		'mp-12829',
		'mp-27811',
		'mp-8180',
		'mp-7971',
		'mp-8449',
		'mp-558045',
		'mp-562394',
		'mp-510073',
		'mp-7744',
		'mp-989562',
		'mp-8892'
	]


	with open(keypath) as fobj:
		apikey = fobj.read().split()[0]
		mpr = MPRester(apikey)
		download(crystals, mpr, dirpath)


def conductivity(sigma, temp):

	lr = LinearRegression()

	y = np.log(sigma).reshape(-1, 1)
	x = np.log(temp).reshape(-1, 1)

	lr.fit(x, y)

	return np.exp(lr.predict(np.log(298).reshape(-1, 1)))


def softbv_call(dirpath, fptr):

	temps = [1250, 1500, 2000]
	ic = []
	sf = str(0.75)
	name = fptr.split('.')
	fpath = osp.join(dirpath, fptr)
	try:
		cube_run = subprocess.run(['softBV.x', '--gen-cube', fpath, 'Li', str(1)], capture_output=True)
		for line in str(cube_run.stdout).split('\\n'):
			split_line = line.split(':')[-1].split(' ')
			if split_line[1:4] == ['final', 'screening', 'factor']:
				sf = split_line[-1]
				break
	except subprocess.TimeoutExpired:
		print('Cube timeout')
	print(fptr, sf)
	cube_path = fpath + '.cube'
	for T in temps:
		try:
			kmc_run = subprocess.run(
				['softBV.x', '--kmc', fpath, cube_path, 'Li', '1', '7', '7', '7', sf, str(T), '0', '100000', '10000', '10'],
				capture_output=True)
			for line in str(kmc_run.stdout).split('\\n'):
				split_line = line.split(' ')
				try:
					if split_line[1] == 'conductivity':
						ic.append(float(split_line[3][:-1]))
						break
				except IndexError:
					continue
		except subprocess.CalledProcessError:
			print(fptr, 'KMC Failed')
		except subprocess.TimeoutExpired:
			print(fptr, 'KMC timeout')

	return ic


def md_run(cif, path_to_pot):
	struc = Structure.from_file(cif)
	print(struc)
	Structure.add_oxidation_state_by_guess(struc)
	name = cif.split('/')[-1].split('.')[0]

	json_fptr = osp.join(path_to_pot, 'm3gnet.json')
	pot_fptr = osp.join(path_to_pot, 'mgnet.index')
	refs = json.load(open('m3gnet.json'))['element_refs']

	model = M3GNet(max_n=3, max_l=3, n_blocks=3, units=64,
	               cutoff=5.0, threebody_cutoff=5.0,
	               include_states=False, readout="weighted_atom",
	               n_atom_types=94, task_type="regression",
	               is_intensive=False, mean=0.0, std=32.17,
	               element_refs=refs)
	
	model.load_weights('m3gnet.index')
	pot = Potential(model=model)
	
	md = MolecularDynamics(
	    atoms=struc,
		potential=pot,
	    temperature=1000,  # 1000 K
	    ensemble='nvt',  # NVT ensemble
	    timestep=2, # 1fs,
	    trajectory=name + ".traj",  # save trajectory to mo.traj
	    logfile=name + ".log",  # log file for MD
	    loginterval=100,  # interval for record the log
	)

	md.run(steps=2500)

@click.command()
@click.option('--dirpath', default='./data')
@click.option('--currpath', default='./')
@click.option('--calc', default='m3gnet')
def run(dirpath, currpath, calc):


	temps = [1250, 1500, 2000]
	count = 0

	for fptr in os.listdir(dirpath)[:]:
		name = fptr.split('.')
		if name[-1] == 'cif' and len(name) == 2:
			if calc=='sbv':
				ic = softbv_call(dirpath, fptr)
				print(ic)
			elif calc=='m3gnet':
				md_run(osp.join(dirpath, fptr), currpath)

		break
		

def verify():
	true = [
		0.07012716,
		3.29437742e-15,
		5.50841659e-06,
		7.41314353e-05,
		0.00144069
	]
	pred = [
		1.4e-2,
		1e-6,
		1e-13,
		1e-7,
		2.5e-4
	]

	plt.plot(np.log(true), np.log(true))
	plt.scatter(np.log(true), np.log(pred), color='k')
	plt.show()


if __name__ == '__main__':
	# data()
	# run()
	T = 1000
	name = "mp-1040450"
	struc = Structure.from_file(name + '.cif')
	Structure.add_oxidation_state_by_guess(struc)
	mol_den = struc.density / struc.composition.weight
	lic = struc.composition.fractional_composition.as_dict()['Li+']
	# for i in range(len(struc)):
	# 	print(i, struc[i])
	traj = Trajectory('mp-1040450.traj')
	calc = DiffusionCoefficient(traj, 200 * units.fs, [0, 1, 2, 3, 4])
	# print(traj[0])
	print(calc.get_diffusion_coefficients()[0][0] * units.fs * 1e-1 * 111.966 * mol_den * lic / T)
