import os
import psi4
import numpy as np
import sf
from sf import fock_ci
from sf import bloch
import numpy as np
import numpy.linalg as LIN

n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 5
H 0 0 0
H 2 0 0
H 0 2 0
H 2 2 0
symmetry c1
""")

options = {"BASIS": "sto-3g", 'e_convergence': 1e-12, 'd_convergence': 1e-12, 'scf_type': 'pk', 'guess': 'gwh', 'reference': 'rohf'}
sf_options = {'SF_DIAG_METHOD': 'LANCZOS', 'RETURN_VECTS': True, 'RETURN_WFN': True, 'NUM_ROOTS': 5}

wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)

np.set_printoptions(threshold=np.inf, linewidth=100000)

# do a 1SF
#e, vects, wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)

H = bloch.do_bloch(wfn, 1, 0, 4, 2.0)

#print(LIN.eig(H))

