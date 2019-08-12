import os
import psi4
import numpy as np
import sf
from sf import fock_ci
from sf import bloch
import heisenberg
import numpy as np
import numpy.linalg as LIN

n2_7 = psi4.core.Molecule.from_string("""
0 5
H 0 0 0
H 2 0 0
H 0 2 0
H 2 2 0
symmetry c1
""")

options = {"BASIS": "6-31G*", 'e_convergence': 1e-12, 'd_convergence': 1e-12, 'scf_type': 'pk', 'guess': 'gwh', 'reference': 'rohf'}
sf_options = {'SF_DIAG_METHOD': 'LANCZOS', 'NUM_ROOTS': 4}

wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)

np.set_printoptions(threshold=np.inf, linewidth=100000)

J = bloch.do_bloch(wfn)

#print(wfn.local_vecs)
#wfn.print_local_dets()

#print("H = -2J * S1 * S2")
#print(219474.63*H)

heis = heisenberg.heis_ham()

heis.do_heisenberg(4, J)

heis.print_roots()

