from __future__ import print_function
"""
Tutorial: A reference implementation of configuration interactions singles.
"""

__authors__   = ["Boyi Zhang", "Adam S. Abbott"]
__credits__   = ["Boyi Zhang", "Adam S. Abbott", "Justin M. Turney"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-08-08"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np

# ==> Set Basic Psi4 Options <==

# Memory specifications
psi4.set_memory(int(2e9))
numpy_memory = 2

# Output options
psi4.core.set_output_file('output.dat', False)

'''
mol = psi4.geometry("""
0 1
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
'''

mol = psi4.geometry("""
        0 3
        N 0 0 0
        N 1.5 0 0
        symmetry c1
        """)

psi4.set_options({'basis':        'cc-pvdz',
                  'scf_type':     'pk',
                  'reference':    'uhf',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# Check memory requirements
nmo = scf_wfn.nmo()
I_size = (nmo**4) * 8e-9
print('\nSize of the ERI tensor will be %4.2f GB.\n' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted \
                     memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Get basis and orbital information
nbf = mints.nbf()          # Number of basis functions
nalpha = scf_wfn.nalpha()  # Number of alpha electrons
nbeta = scf_wfn.nbeta()    # Number of beta electrons
nocc = nalpha + nbeta      # Total number of electrons
nso = 2 * nbf              # Total number of spin orbitals
nvirt = nso - nocc         # Number of virtual orbitals

def spin_block_tei(I):
    '''
    Spin blocks 2-electron integrals
    Using np.kron, we project I and I tranpose into the space of the 2x2 ide
    The result is our 2-electron integral tensor in spin orbital notation
     '''
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)
 
I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)
 
# Convert chemist's notation to physicist's notation, and antisymmetrize
# (pq | rs) ---> <pr | qs>
# <pr||qs> = <pr | qs> - <pr | sq>
gao = I_spinblock.transpose(0, 2, 1, 3) - I_spinblock.transpose(0, 2, 3, 1)

# Get orbital energies, cast into NumPy array, and extend eigenvalues
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)

# Get coefficients, block, and sort
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = np.block([
             [      Ca,         np.zeros_like(Cb)],
             [np.zeros_like(Ca),          Cb     ]])

# Sort the columns of C according to the order of orbital energies
C = C[:, eps.argsort()]

# Sort orbital energies
eps = np.sort(eps)

# Transform gao, which is the spin-blocked 4d array of physicist's notation,
# antisymmetric two-electron integrals, into the MO basis using MO coefficients
gmo = np.einsum('pQRS, pP -> PQRS',
      np.einsum('pqRS, qQ -> pQRS',
      np.einsum('pqrS, rR -> pqRS',
      np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)


# Initialize CIS matrix.
# The dimensions are the number of possible single excitations
HCIS = np.zeros((nocc * nvirt, nocc * nvirt))
print(HCIS.shape)

# Build the possible excitations, collect indices into a list
excitations = []
for i in range(nocc):
    for a in range(nocc, nso):
        excitations.append((i, a))

# Form matrix elements of shifted CIS Hamiltonian
for p, left_excitation in enumerate(excitations):
    i, a = left_excitation
    for q, right_excitation in enumerate(excitations):
        j, b = right_excitation
        HCIS[p, q] = (eps[a] - eps[i]) * (i == j) * (a == b) + gmo[a, j, i, b]

print(HCIS)

# Diagonalize the shifted CIS Hamiltonian
#print("FROM DIAG: ", np.sort(np.linalg.eigvalsh(HCIS)).shape)
print("FROM DIAG: ", scf_e + np.sort(np.linalg.eigvalsh(HCIS)))

