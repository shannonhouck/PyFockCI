from cis import do_sf_cas
import psi4

n2 = psi4.core.Molecule.create_molecule_from_string("""
1 3
H 
H 1 1.0
H 1 1.0 2 60
symmetry c1
""")

print(do_sf_cas(0, 3, n2, conf_space="p", add_opts={'basis': 'sto-3g', 'diis_start': 20, 'e_convergence': 1e-10, 'num_roots': 4, 'scf_type': 'pk', 'd_convergence': 1e-10, 'diag_method': 'rsp'}))
