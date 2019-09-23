import os
import psi4
import numpy as np
import sf_ip_ea
from sf_ip_ea import fock_ci
from sf_ip_ea import bloch
import numpy as np
import numpy.linalg as LIN

n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 4
H 2 0 0
H 0 0 2
H 0 2 0
symmetry c1
""")

options = {"BASIS": "sto-3g", 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'scf_type': 'direct', 'guess': 'gwh', 'reference': 'rohf'}
sf_options = {'SF_DIAG_METHOD': 'LANCZOS', 'NUM_ROOTS': 3}

psi4.set_options(options)
e, psi4_wfn = psi4.energy('scf', molecule=n2_7, return_wfn=True)

ras1 = psi4_wfn.doccpi()[0]
ras2 = psi4_wfn.soccpi()[0]
C = psi4.core.Matrix.to_array(psi4_wfn.Ca())
ras1_C = C[:, :ras1]
ras2_C = C[:, ras1:ras1+ras2]
ras3_C = C[:, ras1+ras2:]
loc = psi4.core.Localizer.build('BOYS', psi4_wfn.basisset(),
          psi4.core.Matrix.from_array(ras2_C))
loc.localize()
U = psi4.core.Matrix.to_array(loc.U, copy=True)
# localize vects
# write localized orbitals to wfn and molden
C_full_loc = psi4.core.Matrix.from_array(np.column_stack((ras1_C,
                 psi4.core.Matrix.to_array(loc.L), ras3_C)))
psi4_wfn.Ca().copy(C_full_loc)
psi4_wfn.Cb().copy(C_full_loc)

sf_options.update({'READ_PSI4_WFN': True, 'PSI4_WFN': psi4_wfn})
wfn = fock_ci( 1, 1, n2_7, conf_space="neutral", ref_opts=options, sf_opts=sf_options)

#wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)

ras2 = 3
newvecs = np.zeros((ras2, ras2, sf_options['NUM_ROOTS']))
for i in range(ras2):
    for n in range(sf_options['NUM_ROOTS']):
        newvecs[i, i, n] = wfn.vecs[i, n]
newvecs = np.reshape(newvecs, (ras2*ras2, sf_options['NUM_ROOTS']))

wfn.vecs = newvecs

np.set_printoptions(threshold=np.inf, linewidth=100000)
J = bloch.do_bloch(wfn, 3, skip_localization=True)

np.set_printoptions(precision=12)
print("J")
print(J)

#heis = heisenberg.heis_ham()
#heis.do_heisenberg([1,1,1], J)
#heis.print_roots()


