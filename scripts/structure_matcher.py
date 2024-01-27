from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

malnac_lattice = Lattice.from_parameters(5.33, 5.14, 11.25, 102.7, 135.17, 85.17)
malnac02_lattice = Lattice.from_parameters(5.156, 5.341, 8.407, 71.48, 76.12, 85.09)

# here, because we're only interested in the lattices,
# we can define a "dummy structure" with a single H atom at the origin

malnac_struct = Structure(malnac_lattice, ["H"], [[0.0, 0.0, 0.0]])
malnac02_struct = Structure(malnac02_lattice, ["H"], [[0.0, 0.0, 0.0]])

# StructureMatcher can accept different tolerances for judging equivalence
matcher = StructureMatcher(primitive_cell=False)

# first, we can verify these lattices are equivalent
matcher.fit(malnac_struct, malnac02_struct)  # returns True

# and we can get the transformation matrix from one to the other
# this returns the supercell matrix (e.g. change of basis),
# as well as any relevant translation, and mapping of atoms from one
# crystal to the other
matcher.get_transformation(malnac_struct, malnac02_struct)
# returns (array([[ 0, -1,  0], [-1,  0,  0], [ 0,  1,  1]]), array([0., 0., 0.]), [0])

# Compute the distance between both
matcher.get_rms_dist(malnac_struct, malnac02_struct)[0]

# Create a structure with pytmatgen
l = Lattice.orthorhombic(1, 2, 12)  # lattice
sp = ["Si", "Si", "Al"]  # elements
s1 = Structure(l, sp, [[0.5, 0, 0], [0, 0, 0], [0, 0, 0.5]])  # position of atoms
s2 = Structure(l, sp, [[0.5, 0, 0], [0, 0, 0], [0, 0, 0.6]])
matcher.fit(s1, s2)  # False
matcher.get_rms_dist(s1, s2)[0]  # 0.282
