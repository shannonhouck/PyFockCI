import sys, os
import psi4
sys.path.insert(1, '../')
sys.path.insert(1, '../../')
import spinflip
from spinflip import sf_cas as sf_cas_ref
import sf
from sf import fock_ci
from sf import post_ci_analysis
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

'''
# Test: 1SF-CAS
def test_cas_1sf():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [6.0, 12.0, 6.0, 6.0, 6.0, 6.0]
    e, vects = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 0, "", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(h)-1SF
def test_h_1sf():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [6.0, 12.0, 6.0, 6.0, 6.0, 6.0]
    e, vects = fock_ci( 1, 1, n2_7, conf_space="h", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 0, "h", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(p)-1SF
def test_p_1sf():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [6.0, 12.0, 6.0, 6.0, 6.0, 6.0]
    e, vects = fock_ci( 1, 1, n2_7, conf_space="p", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 0, "p", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(hp)-1SF
def test_hp_1sf():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [6.0, 12.0, 6.0, 6.0, 6.0, 6.0]
    e, vects = fock_ci( 1, 1, n2_7, conf_space="h,p", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 0, "h,p", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold

# Test: EA
def test_cas_ea():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 0, 1, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, 1, "", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: IP
def test_cas_ip():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 1, 0, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, -1, "", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: 1SF-EA-CAS
def test_cas_1sfea():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 3.75, 3.75, 3.75, 3.75, 3.75]
    e, vects = fock_ci( 1, 2, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 1, "", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: 1SF-IP-CAS
def test_cas_1sfip():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 3.75, 3.75, 3.75, 8.75, 8.75]
    e, vects = fock_ci( 2, 1, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, -1, "", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(h)-1SF-IP
def test_h_1sfip():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 3.75, 3.75, 3.75, 8.75, 8.75]
    e, vects = fock_ci( 2, 1, n2_7, conf_space="h", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, -1, "h", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold
'''

# Test: RAS(p)-1SF-EA
def test_p_1sfea():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 3.75, 3.75, 3.75, 3.75, 3.75]
    e, vects = fock_ci( 1, 2, n2_7, conf_space="p", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(1, 1, "p", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

'''
# Test: RAS(h)-EA
def test_h_ea():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 0, 1, n2_7, conf_space="h", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, 1, "h", vects[:, i], 4, 6, 18)
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(h)-IP
def test_h_ip():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 1, 0, n2_7, conf_space="h", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, -1, "h", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(p)-EA
def test_p_ea():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 0, 1, n2_7, conf_space="p", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, 1, "p", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: RAS(p)-IP
def test_p_ip():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [8.75, 8.75, 8.75, 8.75, 8.75]
    e, vects = fock_ci( 1, 0, n2_7, conf_space="p", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(0, -1, "p", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold

# Test: 2SF-CAS
def test_cas_2sf():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}
    s_expected = [2.0, 6.0, 12.0, 6.0, 2.0, 2.0]
    e, vects = fock_ci( 2, 2, n2_7, conf_space="", ref_opts=options, sf_opts={'return_vects': True, 'sf_diag_type': 'LANCZOS'} )
    for i in range(len(s_expected)):
        s2 = post_ci_analysis.calc_s_squared(2, 0, "", vects[:, i], 4, 6, 18) 
        assert abs(s2 - s_expected[i]) < threshold
'''
