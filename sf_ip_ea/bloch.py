from __future__ import print_function
import numpy as np
import math
import psi4
import scipy.linalg as LIN

def lowdin_orth(A):
    """
    Performs Lowdin orthonormalization on a given matrix A.

    Orthonormalizes a given matrix A based on Lowdin's approach.

    Parameters:
        A (numpy.ndarray): NumPy array to orthogonalize.

    Returns:
        numpy.ndarray: An orthogonalized version of A.
    """
    U, S, V = LIN.svd(A)
    return np.dot(U, V)

def do_bloch(wfn, n_sites, site_list=None, site_list_orbs=None,
    molden_file='orbs.molden', skip_localization=False, neutral=False):
    """
    Bloch effective Hamiltonian solver.

    Solves the Bloch effective Hamiltonian and returns a matrix of the J couplings.
    It is totally dependent on Psi4 for localization right now.

    Parameters:
        wfn (wfn): SF-IP-EA wfn object containing info about the calculation.
        n_sites (int): The number of sites (integer).
        site_list (list): List of which atoms are "sites". If this is not given,
            the program assumes one orbital per site.
        molden_file (str): Name of the Molden file to write for localized orbitals.
        skip_localization (bool): Whether to skip orbital localization. If you use
            this, you should localize the orbitals in wfn.wfn yourself!
    
    Returns:
        J (numpy.ndarray): NumPy matrix of J coupling values
    """

    np.set_printoptions(suppress=True)
    print("Doing Bloch Hamiltonian analysis...")

    # Put input vector into block form (only need CAS-1SF block!!)
    n_SF = wfn.n_SF
    ras1 = wfn.ras1
    ras2 = wfn.ras2
    e = wfn.e.copy()
    vecs = wfn.vecs
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

    print(v_b1)

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
                perm.append(orb - ras1)
        print("Reordering RAS2 determinants as follows:")
        print(perm)
        v_n = v_n[np.argsort(perm), :] # permute!
        # construct coeff matrix
        R = np.zeros((ras2, len(site_list_orbs)))
        for i, site in enumerate(np.sort(perm)):
            R[i, site] = 1.0/math.sqrt(len(site_list_orbs[i]))
        # orthonormalize (SVD)
        v_n = np.dot(R.T, v_n)

    # Else, assume 1 orbital per site
    else:
        orbs_per_site = n_sites*[1]

    #v_orth = v_n
    v_orth = lowdin_orth(v_n)

    print("v_n")
    print(v_n)

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
    print("J Couplings:")
    for i in range(n_sites):
        for j in range(i):
            Sa = orbs_per_site[i]/2.0
            Sb = orbs_per_site[j]/2.0
            J[i,j] = J[j,i] = -1.0*H[i,j]/(2.0*math.sqrt(Sa*Sb))
            print("\tJ%i%i = %6.6f" %(i, j, J[i,j]))

    return J


