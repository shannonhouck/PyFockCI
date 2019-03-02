import sys, os
import psi4
sys.path.insert(1, '../')
sys.path.insert(1, '../../')
import spinflip
from spinflip import sf_cas as sf_cas_ref
import sf
from sf import sf_psi4
import time

# threshold for value equality
threshold = 1e-7
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
    time.sleep(0.1)
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = sf_cas_ref( 0, 5, n2_7, conf_space="1x", add_opts=options, num_roots=6 )
    e = sf_psi4( 1, 1, n2_7, conf_space="h,p", add_opts=options )
    assert abs(e[0] - expected) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    time.sleep(0.1)
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'calc_s_squared': True, 'scf_type': 'pk'}
    expected = sf_cas_ref( 0, 3, n2_3, conf_space="1x", add_opts=options, num_roots=6 )
    e = sf_psi4( 1, 1, n2_3, conf_space="h,p", add_opts=options )
    assert abs(e[0] - expected) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    time.sleep(0.1)
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = sf_cas_ref( 0, 1, o2, conf_space="1x", add_opts=options )
    e = sf_psi4( 1, 1, o2, conf_space="h,p", add_opts=options, num_roots=2 )
    assert abs(e[0] - expected) < threshold
