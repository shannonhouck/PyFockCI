"""
Bloch effective Hamiltonian solver.

This module constructs a Bloch effective Hamiltonian for a given SF-IP/EA 
wavefunction. Note that this module is currently dependent on Psi4.

"""

from __future__ import print_function
import numpy as np
import math
import psi4
import scipy.linalg as LIN

def lowdin_orth(A):
    """
    Performs Lowdin orthonormalization on a given matrix A.

    Orthonormalizes a given matrix A based on Lowdin's approach.

    :param A: NumPy array to orthogonalize.
    :return: An orthogonalized version of A.
    """
    U, S, V = LIN.svd(A)
    return np.dot(U, V)

def do_bloch(wfn, n_sites, site_list=None, site_list_orbs=None,
    molden_file='orbs.molden', skip_localization=False, neutral=False):
    """
    Bloch effective Hamiltonian solver.

    Solves the Bloch effective Hamiltonian and returns a matrix containing 
    J coupling information. Sites are designated using ``site_list`` or 
    ``site_list_orbs``; if neither of these keywords are specified, 
    each orbital in the CAS/RAS2 space is assumed to be its own site. 
    A Molden file containing the orbitals is written to ``orbs.molden`` (or 
    whichever file is specified by the user). The J coupling values are 
    also written to standard output.

    Parameters
    ----------
    wfn (sf_wfn): SF-IP-EA wfn object containing info about the calculation.
    n_sites (int) : The number of sites.
    site_list (list) : List of which atoms are "sites". If this is not given,
        the program assumes one orbital per site. Atomic center ordering
        starts at zero. Optional.
    site_list_orbs (list): A list of sites, using lists of orbitals rather 
        than atomic centers. (It's a list of lists. For example, for a 
        two-site case where MOs 55, 56, and 59 are on one site and the 
        remaining orbitals are on the other, use ``[[55,56,59],[57,58,60]]``.) 
        Note that ordering starts at 1, not zero, so it follows the same 
        indexing as the MO printing in the Psi4 output files. Optional.
    molden_file (string) : Molden filename to which orbitals are written. 
        Optional. Defaults to ``orbs.molden``.
    skip_localization (bool) : Whether to skip orbital localization. If true, 
        the user should localize the orbitals in wfn.wfn beforehand! 
        Optional. Defaults to False.
    neutral (bool) : Calculate the J couplings using neutral determinants 
        only. Optional. Defaults to False.

    Returns
    -------
    NumPy matrix of J coupling values
    """

    np.set_printoptions(suppress=True)
    print("Doing Bloch Hamiltonian analysis...")

    # Put input vector into block form (only need CAS-1SF block!!)
    n_SF = wfn.n_SF
    ras1 = wfn.ras1
    ras2 = wfn.ras2
    e = wfn.e.copy()
    vecs = wfn.vecs
    # special case for unpacking neutral determinant space
    if(neutral):
        newvecs = np.zeros((ras2, ras2, n_sites))
        for i in range(ras2):
            for n in range(n_sites):
                newvecs[i, i, n] = wfn.vecs[i, n]
        newvecs = np.reshape(newvecs, (ras2*ras2, n_sites))
        vecs = newvecs
    v_b1 = vecs[:(ras2*ras2), :n_sites].copy()
    n_roots = v_b1.shape[1]
    v_b1 = np.reshape(v_b1, (ras2,ras2,n_roots)) # v[i,a]

    # Obtain info for orbital localization and localize v
    psi4_wfn = wfn.wfn
    C = psi4.core.Matrix.to_array(psi4_wfn.Ca(), copy=True)
    # Allow user to skip our localization and use their own if needed
    if(not skip_localization):
        ras1_C = C[:, :ras1]
        ras2_C = C[:, ras1:ras1+ras2]
        ras3_C = C[:, ras1+ras2:]
        loc = psi4.core.Localizer.build('BOYS', psi4_wfn.basisset(), 
                  psi4.core.Matrix.from_array(ras2_C))
        loc.localize()
        U = psi4.core.Matrix.to_array(loc.U, copy=True)
        # localize vects
        v_b1 = np.einsum("ji,jbn->ibn", U, v_b1)
        v_b1 = np.einsum("ba,ibn->ian", U, v_b1)
        wfn.local_vecs = np.reshape(v_b1, (ras2*ras2, n_roots))
        # write localized orbitals to wfn and molden
        C_full_loc = psi4.core.Matrix.from_array(np.column_stack((ras1_C, 
                         psi4.core.Matrix.to_array(loc.L), ras3_C)))
        psi4_wfn.Ca().copy(C_full_loc)
        psi4_wfn.Cb().copy(C_full_loc)

    # write Molden file
    psi4.molden(psi4_wfn, molden_file)

    # Extract i=a part (neutral determinants only!!)
    v_n = None
    for i in range(v_b1.shape[2]):
        v_new = np.diagonal(v_b1[:, :, i])
        if(type(v_n) == type(None)):
            v_n = v_new
        else:
            v_n = np.vstack((v_n, v_new))
    v_n = v_n.T # make sure columns are states rather than rows

    # Obtain S
    # S should be I if states are orthonormal

    orbs_per_site = []
    # Handle grouping orbitals if needed
    if(type(site_list) != type(None)):
        # Construct density N for sites
        N = np.zeros((len(site_list), ras2))
        bas = psi4_wfn.basisset()
        S = psi4.core.Matrix.to_array(psi4_wfn.S())
        C = psi4.core.Matrix.to_array(psi4_wfn.Ca())
        CS = np.einsum("vi,vu->ui", C, S)
        for atom, A in enumerate(site_list):
            for i in range(ras2):
                for mu in range(C.shape[1]):
                    if(bas.function_to_center(mu) == A):
                        N[atom, i] += C[mu, ras1+i] * CS[mu, ras1+i]
        # Reorder v_n rows so they're grouped by site
        perm = []
        for i in range(ras2):
            diff = abs(N[:, i]-1)
            perm.append(np.argmin(diff))
        print("Reordering RAS2 determinants as follows:")
        print(perm)
        v_n = v_n[np.argsort(perm), :] # permute!
        # construct coeff matrix
        R = np.zeros((ras2, len(site_list)))
        tmp, orbs_per_site = np.unique(perm, return_counts=True)
        for i, site in enumerate(np.sort(perm)):
            R[i, site] = 1.0/math.sqrt(orbs_per_site[site])
        # orthonormalize (SVD)
        v_n = np.dot(R.T, v_n)

    elif(type(site_list_orbs) != type(None)):
        # Reorder v_n rows so they're grouped by site
        perm = []
        for site in site_list_orbs:
            for orb in site:
                perm.append(orb - ras1 - 1)
        print("Reordering RAS2 determinants as follows:")
        print(perm)
        v_n = v_n[perm, :] # permute!
        # construct coeff matrix
        # construct coeff matrix
        R = np.zeros((ras2, len(site_list_orbs)))
        ind = 0
        for s, site in enumerate(site_list_orbs):
            orbs_per_site.append(len(site))
            for i in range(len(site)):
                R[ind, s] = 1.0/math.sqrt(len(site))
                ind = ind + 1
        # orthonormalize (SVD)
        v_n = np.dot(R.T, v_n)

    # Else, assume 1 orbital per site
    else:
        orbs_per_site = n_sites*[1]

    #v_orth = v_n
    v_orth = lowdin_orth(v_n)

    S = np.dot(v_orth.T, v_orth)
    print("Orthogonalized Orbital Overlap:")
    print(S)

    # Build Bloch Hamiltonian
    #H = np.dot(S, v_orth)
    H = v_orth
    H = np.dot(H, np.diag(e))
    H = np.dot(H, v_orth.T) # invert v_orth
    J = np.zeros(H.shape)
    print("Effective Hamiltonian")
    print(H)
    print("Orbs Per Site")
    print(orbs_per_site)
    print("J Couplings:")
    for i in range(n_sites):
        for j in range(i):
            Sa = orbs_per_site[i]/2.0
            Sb = orbs_per_site[j]/2.0
            J[i,j] = J[j,i] = -1.0*H[i,j]/(2.0*math.sqrt(Sa*Sb))
            print("\tJ%i%i = %6.6f" %(i, j, J[i,j]))

    return J


