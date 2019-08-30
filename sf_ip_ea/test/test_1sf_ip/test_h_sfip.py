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

# Test: 1SF-CAS
@pytest.mark.methodtest
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = [-108.257015341634414, -108.250447679154192, -108.231958280565408, -108.231958280565280]
    wfn = sf_ip_ea.fock_ci( 2, 1, n2_7, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

@pytest.mark.methodtest
def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = [-108.344924167201228, -108.310124617639175, -108.310124617639104, -108.299904314302879]
    wfn = sf_ip_ea.fock_ci( 2, 1, n2_3, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 4, 'SF_DIAG_METHOD': 'LANCZOS'} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

@pytest.mark.methodtest
def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    expected = [-149.207583364661616, -149.171950340935524]
    wfn = sf_ip_ea.fock_ci( 2, 1, o2, conf_space="h", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

