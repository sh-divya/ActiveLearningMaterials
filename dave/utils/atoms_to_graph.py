from itertools import product

import ase
import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import pymatgen
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from torch_geometric.data import Data
from torch_scatter import segment_coo, segment_csr
from tqdm import tqdm


def compute_neighbors(data, edge_index):
    if isinstance(data.natoms, int):
        data.natoms = torch.LongTensor([data.natoms])
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


def collate(data_list):
    # from ocpmodels.commom.utils
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
        elif isinstance(item[key], int) or isinstance(item[key], float):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


class AtomsToGraphs:
    # from ocpmodels/preprocessing/atoms_to_graphs.py
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.

    """

    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_fixed=True,
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        if not isinstance(atoms, pymatgen.core.structure.Structure):
            struct = AseAtomsAdaptor.get_structure(atoms)
        else:
            struct = atoms
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(
        self,
        atoms,
    ):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
        )

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list


def make_a2g():
    """ Convert periodic atomic structures to graphs """
    return AtomsToGraphs(
        max_neigh=50,
        radius=6.0,
        r_energy=False,
        r_forces=False,
        r_distances=True,
        r_edges=True,
    )


def pymatgen_structure_to_graph(struct, a2g: AtomsToGraphs = None):
    if a2g is None:
        a2g = make_a2g()
    atoms = AseAtomsAdaptor.get_atoms(struct)
    data = a2g.convert(atoms)
    return data


def pymatgen_struct_to_pyxtal_to_graphs(struct, a2g=None, to_conventional=True, n=1):
    # %timeit pymatgen_struct_to_pyxtal_to_graph(Structure.from_str(data_df.sample().iloc[0]["cif"], fmt="cif"))
    # 279 ms ± 178 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    if a2g is None:
        a2g = make_a2g()
    if to_conventional:
        SGA = SpacegroupAnalyzer(struct)
        struct = SGA.get_conventional_standard_structure()
    spg: int = struct.get_space_group_info()[1]
    comp: dict = struct.composition.get_el_amt_dict()
    ltype: str = SpacegroupAnalyzer(struct).get_crystal_system()
    lattice = Lattice.from_para(
        *struct.lattice.abc, *struct.lattice.angles, ltpye=ltype
    )
    results = []
    for _ in range(n):
        s = pyxtal()
        s.from_random(
            3,
            spg,
            list(comp.keys()),  # Species strings
            list(comp.values()),  # Species counts
            lattice=lattice,
            conventional=to_conventional,
        )
        results.append(pymatgen_structure_to_graph(s.to_pymatgen(), a2g))
    return results


if __name__ == "__main__":
    # Crystal positions sampling
    cell = Lattice.from_para(7.8758, 7.9794, 5.6139, 90, 90, 90, ltype="orthorhombic")
    spg = 58
    elements = ["Al", "Si", "O"]
    composition = [8, 4, 20]

    sites = [
        {
            "4e": [0.0000, 0.0000, 0.2418],
            "4g": [0.1294, 0.6392, 0.0000],
        },
        {"4g": [0.2458, 0.2522, 0.0000]},
        {"4g": [0.4241, 0.3636, 0.0000]},  # partial information on O sites
    ]

    s = pyxtal()
    s.from_random(3, spg, elements, composition, lattice=cell, sites=sites)
    print(s)
    struct = s.to_pymatgen()

    a2g = make_a2g()

    data = pymatgen_structure_to_graph(struct, a2g)

    print()
    print(data)
