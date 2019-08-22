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
0 5
O
O 1 1.2
symmetry c1
""")

# Test: 1SF-CAS
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    expected = [-108.774549178387815, -108.771464507398591, -108.766379708729502, -108.672439123596263]
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    wfn = sf_ip_ea.fock_ci( 2, 2, n2_7, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    expected = [-108.845616548841292, -108.779860044401957, -108.717563861488429, -108.717563861488415]
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'direct'}
    wfn = sf_ip_ea.fock_ci( 2, 2, n2_3, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 4} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

def test_3():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    expected = [-149.594136747555211, -149.592058111435961]
    options = {"basis": "cc-pvdz", 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    wfn = sf_ip_ea.fock_ci( 2, 2, o2, conf_space="", ref_opts=options, sf_opts={'NUM_ROOTS': 2} )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

