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
    expected = [-108.250639578975210, -108.216113001125137, -108.216113001125137, -108.194918023564696]
    wfn = sf_ip_ea.fock_ci( 1, 0, n2_7, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'calc_s_squared': True, 'scf_type': 'pk'}
    expected = [-108.230235021686696, -108.230235021686696, -108.042026971032641, -108.042026971032641]
    wfn = sf_ip_ea.fock_ci( 1, 0, n2_3, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = [-149.083861036267706, -149.083861036267706]
    wfn = sf_ip_ea.fock_ci( 1, 0, o2, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

