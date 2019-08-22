import sys, os
import psi4
import sf_ip_ea

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
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = [-108.257015341635110, -108.220329170163453, -108.220329170162785, -108.199753179193195]
    wfn = sf_ip_ea.fock_ci( 1, 0, n2_7, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'calc_s_squared': True}
    expected = [-108.299904314302950, -108.247296829002906, -108.236785088491118, -108.236785088490976]
    wfn = sf_ip_ea.fock_ci( 1, 0, n2_3, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = [-149.083861036268075, -149.083861036267962]
    wfn = sf_ip_ea.fock_ci( 1, 0, o2, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

