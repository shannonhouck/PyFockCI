import psi4
import sf_ip_ea
import pytest

# threshold for value equality
threshold = 1e-7
# setting up molecule
n2_7 = psi4.core.Molecule.from_string("""
0 7
N 0 0 0
N 0 0 1.3
symmetry c1
""")

# Test: 1SF-CAS with frozen core/virtual
@pytest.mark.frozentest
def test_1():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    options = {'basis': 'cc-pvdz', 'e_convergence': 1e-10, 'd_convergence': 1e-10,
               'scf_type': 'direct'}
    expected = [
        -108.696835145327,
        -108.662316272217,
        -108.662316272217,
        -108.480005065603,
        -108.480005065603,
        -108.438584969439,
        -108.362904257363]
    sf_options = {'SF_DIAG_METHOD': 'DAVIDSON',
              'NUM_ROOTS': 7, 'INTEGRAL_TYPE': 'full',
              'frozen_virt': 5, 'frozen_core': 3}
    wfn = sf_ip_ea.fock_ci( 1, 1, n2_7, conf_space="h,p", ref_opts=options, sf_opts=sf_options )
    for i, true in enumerate(wfn.e):
        assert abs(true - expected[i]) < threshold

