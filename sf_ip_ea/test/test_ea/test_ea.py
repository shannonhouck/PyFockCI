import sys, os
import psi4
import sf_ip_ea

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
    expected = [-108.600832070120418, -108.566664526889454, -108.566664526889440, -108.533875094978811]
    wfn = sf_ip_ea.fock_ci( 0, 1, n2_7, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'calc_s_squared': True, 'scf_type': 'pk'}
    expected = [-108.630935262330539, -108.630935262330468, -108.431711959124172, -108.431711959124172]
    wfn = sf_ip_ea.fock_ci( 0, 1, n2_3, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'diag_method': 'rsp'}
    expected = [-149.491801512701585, -149.491801512701585]
    wfn = sf_ip_ea.fock_ci( 0, 1, o2, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

