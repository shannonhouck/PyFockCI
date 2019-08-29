import psi4
import sf_ip_ea
from sf_ip_ea import bloch
import numpy as np
import time

# threshold for value equality
threshold = 1e-7
# setting up molecule
h4 = psi4.core.Molecule.from_string("""
0 5
H 0 0 0
H 2 0 0
H 0 2 0
H 2 2 0
symmetry c1
""")
o3 = psi4.core.Molecule.from_string("""
0 7
O 0.0 0.0 0.0
O 2.0 0.0 0.0
O 4.0 0.0 0.0
symmetry c1
""")

# Test: 1SF-CAS
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "6-31G*", 'num_roots': 4, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'guess': 'gwh'}
    sf_opts = {"NUM_ROOTS": 4}
    J_ref = np.array([[ 0.0, -0.01307327091, -0.01307327091, 0.000741724797],
                      [-0.01307327091, 0.0, 0.000741724797, -0.01307327091 ],
                      [-0.01307327091, 0.000741724797, 0.0, -0.01307327091 ],
                      [ 0.000741724797, -0.01307327091, -0.01307327091, 0.0]] )
    wfn = sf_ip_ea.fock_ci(1, 1, h4, conf_space="", ref_opts = options, sf_opts=sf_opts)
    J = bloch.do_bloch(wfn, 4)
    assert np.allclose(J, J_ref)

def test_2():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "6-31G*", 'num_roots': 3, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'direct', 'guess': 'gwh'}
    sf_opts = {"NUM_ROOTS": 3}
    J_ref = np.array([[ 0.0, 0.001437732854, -0.000302643717],
                      [ 0.001437732854, 0.0, 0.001437732854],
                      [-0.000302643717, 0.001437732854, 0.0]] )
    wfn = sf_ip_ea.fock_ci(1, 1, o3, conf_space="", ref_opts = options, sf_opts=sf_opts)
    J = bloch.do_bloch(wfn, 3, site_list=[0, 1, 2])
    assert np.allclose(J, J_ref)

