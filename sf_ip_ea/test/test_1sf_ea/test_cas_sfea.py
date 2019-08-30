import psi4
import sf_ip_ea
import pytest

# threshold for value equality
threshold = 1e-7
# setting up molecule
n2_7 = psi4.core.Molecule.from_string("""
0 7
N 0 0 0
N 0 0 2.5
symmetry c1
""")
n2_3 = psi4.core.Molecule.from_string("""
0 5
N 0 0 0
N 0 0 1.5
symmetry c1
""")
o2 = psi4.core.Molecule.from_string("""
0 5
O
O 1 1.2
symmetry c1
""")

# Test: 1SF-CAS-EA
@pytest.mark.methodtest
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = [-108.600832070120390, -108.592328021131834, -108.589663624526210, -108.589663624526196]
    wfn = sf_ip_ea.fock_ci( 1, 2, n2_7, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

@pytest.mark.methodtest
def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'calc_s_squared': True, 'scf_type': 'pk'}
    expected = [-108.715101659331566, -108.715101659331509, -108.630935262539580, -108.630935262539552]
    wfn = sf_ip_ea.fock_ci( 1, 2, n2_3, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

@pytest.mark.methodtest
def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = [-149.480852574520867, -149.476508967884286]
    wfn = sf_ip_ea.fock_ci( 1, 2, o2, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

