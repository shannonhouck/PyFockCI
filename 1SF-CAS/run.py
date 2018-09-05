from cis import do_sf_cas
import psi4

n2 = psi4.core.Molecule.create_molecule_from_string("""
0 5
N 0 0 0
N 1.5 0 0
symmetry c1
""")

print(do_sf_cas(0, 3, n2, conf_space="h", add_opts={'basis': 'cc-pvdz', 'diis_start': 20, 'e_convergence': 1e-10, 'num_roots': 4, 'scf_type': 'pk', 'd_convergence': 1e-10, 'diag_method': 'rsp'}))
