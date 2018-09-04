from cis import do_sf_cas
import psi4

n2 = psi4.core.Molecule.create_molecule_from_string("""
0 7
N 0 0 0
N 1.5 0 0
symmetry c1
""")

print(do_sf_cas(0, 5, n2))
