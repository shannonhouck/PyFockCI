import os
import psi4
import numpy as np
import sf
from sf import fock_ci
from sf import bloch
import heisenberg
import numpy as np
import numpy.linalg as LIN

n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 7
O
O 1 3.0
O 1 3.0 2 116.8
symmetry c1
""")
"""
# 1.278
O
O 1 3.0
O 1 3.0 2 116.8
H 0 0 0
H 2 0 0
H 0 2 0
H 2 2 0
"""

options = {"BASIS": "sto-3g", 'e_convergence': 1e-12, 'd_convergence': 1e-12, 'scf_type': 'pk', 'guess': 'gwh', 'reference': 'rohf'}
sf_options = {'SF_DIAG_METHOD': 'LANCZOS', 'NUM_ROOTS': 6}

print("***** TEST: NO READ PSI WFN")
wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)

"""
print("***** TEST: READ PSI WFN")
psi4.core.clean()

psi4.set_options(options)
e, rohf_wfn = psi4.energy('scf', molecule=n2_7, return_wfn=True)

sf_options.update({'READ_PSI4_WFN': True,
                   'PSI4_WFN': rohf_wfn})

wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)
"""

#psi4.molden(wfn.wfn, 'o3.molden')
#psi4.oeprop(wfn.wfn, 'MULLIKEN_CHARGES')

np.set_printoptions(threshold=np.inf, linewidth=100000)
J = bloch.do_bloch(wfn, [0, 1, 2])

#print(wfn.local_vecs)
#wfn.print_local_dets()
#print("H = -2J * S1 * S2")
#print(219474.63*H)
#heis = heisenberg.heis_ham()
#heis.do_heisenberg(4, J)
#heis.print_roots()

