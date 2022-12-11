from sklearn.linear_model import LinearRegression
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import subprocess
import click
import os


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
	crystals = [
		"Li5PS4Cl2",
		"Li6PS5Cl",
		"Li3PS4",
		# "LiYCl6",
		"Li10GeP2S12",
		"Li7La3Zr2O12",
		"Li3ScCl6"
		# "Li3HoCl6",
		# "Li3ScBr6",
		# "Li3HoBr6",
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


def kmc_call(dirpath, fptr):

	temps = [1250, 1500, 2000]
	ic = []
	sf = str(0.75)
	name = fptr.split('.')
	fpath = osp.join(dirpath, fptr)
	try:
		cube_run = subprocess.run(['softBV.x', '--gen-cube', fpath, 'Li', str(1)], capture_output=True)
		print('Here')
		for line in str(cube_run.stdout).split('\\n'):
			split_line = line.split(':')[-1].split(' ')
			print(split_line)
			# print(split_line)
			if split_line[1:4] == ['final', 'screening', 'factor']:
				sf = split_line[-1]
				break
	except subprocess.TimeoutExpired:
		print('Cube timeout')
	print(fptr, sf)
	cube_path = fpath + '.cube'
	for T in temps:
		# cmd = ' '.join(['softBV.x', '--kmc', fpath, cube_path, 'Li', '1', '5 5 5', sf, str(T), '0', '10000', '10000', '10'])
		# print(cmd)
		try:
			kmc_run = subprocess.run(
				['softBV.x', '--kmc', fpath, cube_path, 'Li', '1', '10', '10', '10', sf, str(T), '0', '100000', '10000', '10'],
				capture_output=True, timeout=2 * 60 * 3600)
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


@click.command()
@click.option('--dirpath', default='./data')
@click.option('--outpath', default='./tmp')
def run(dirpath, outpath):

	if not osp.isdir(outpath):
		os.makedirs(outpath)

	temps = [1250, 1500, 2000]

	for fptr in os.listdir(dirpath)[:]:	
		name = fptr.split('.')
		if name[-1] == 'cif' and len(name) == 2:
			ic = kmc_call(dirpath, fptr)
			if ic:
				print(conductivity(ic, temp))
			else:
				print(fptr, 'no ic')
		

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
	run()
