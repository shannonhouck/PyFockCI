import psi4
import sf_ip_ea
from sf_ip_ea import fock_ci
import time

# threshold for value equality
threshold = 1e-7
# setting up molecule
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
    sf_opts = {"NUM_ROOTS": 4}
    expected = [-108.771464507401, -108.766379708730, -108.672439123601, -108.667055270021]
    wfn = fock_ci(1, 1, n2_7, conf_space="", ref_opts = options, sf_opts=sf_opts)
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'direct'}
    sf_opts = {"NUM_ROOTS": 4}
    expected = [-108.779860044370, -108.717563861440, -108.717563861440, -108.711324264599]
    wfn = fock_ci(1, 1, n2_3, conf_space="", ref_opts = options, sf_opts=sf_opts)
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = [-149.609461120738757, -149.561899036996437]
    wfn = fock_ci(1, 1, o2, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 2})
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

