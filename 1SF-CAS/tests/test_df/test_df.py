import sys, os
import psi4
sys.path.insert(1, '../')
import spinflip
from spinflip import sf_cas as sf_cas_ref
import sf
from sf import do_sf_cas

# threshold for value equality
threshold = 1e-7
# setting up molecule
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
    options = {"basis": "cc-pvdz", 'num_roots': 4, 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'df', 'reference': 'rohf'}
    psi4.set_options(options)
    expected = psi4.energy('scf', molecule=o2)
    e = do_sf_cas( 1, 1, o2, conf_space="", add_opts=options, integral_type="DF" )
    assert abs(e - expected) < threshold

