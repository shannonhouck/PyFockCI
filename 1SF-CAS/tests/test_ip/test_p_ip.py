import sys, os
import psi4
sys.path.insert(1, '../')
sys.path.insert(1, '../../')
import spinflip
from spinflip import sf_cas as sf_cas_ref
import sf
from sf import fock_ci

# threshold for value equality
threshold = 1e-7
# setting up molecule
h3 = psi4.core.Molecule.create_molecule_from_string("""
0 4
H 
H 1 1.0 
H 1 1.0 2 60.0
symmetry c1
""")
n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 7
N 0 0 0
N 0 0 2.5
symmetry c1
""")
n2_3 = psi4.core.Molecule.create_molecule_from_string("""
0 5
N 0 0 0
N 0 0 1.5
symmetry c1
""")
o2 = psi4.core.Molecule.create_molecule_from_string("""
0 3
O
O 1 1.2
symmetry c1
""")

# Test: 1SF-CAS
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'pk', 'diag_method':'rsp'}
    expected = sf_cas_ref( 1, 6, n2_7, conf_space="p", add_opts=options )
    e = fock_ci( 1, 0, n2_7, conf_space="p", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    assert abs(e[0] - expected) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'pk', 'diag_method': 'rsp'}
    expected = sf_cas_ref( 1, 4, n2_3, conf_space="p", add_opts=options )
    e = fock_ci( 1, 0, n2_3, conf_space="p", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    assert abs(e[0] - expected) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp', 'scf_type': 'pk'}
    expected = sf_cas_ref( 1, 2, o2, conf_space="p", add_opts=options )
    e = fock_ci( 1, 0, o2, conf_space="p", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    assert abs(e[0] - expected) < threshold

