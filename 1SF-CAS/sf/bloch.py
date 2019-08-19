from __future__ import print_function
import numpy as np
import psi4
import scipy.linalg as LIN

def lowdin_orth(A):
    U, S, V = LIN.svd(A)
    return np.dot(U, V)

def lowdin_orth_2(A):
    sal, svec = np.linalg.eigh(np.dot(A.T, A))
    idx = sal.argsort()[::-1]                         
    sal = sal[idx]                                    
    svec = svec[:, idx]                               
    sal = sal**-0.5                                   
    sal = np.diagflat(sal)                            
    X = svec.dot(sal.dot(svec.T))   
    return np.dot(A, X)

def do_bloch(wfn, site_list, molden_file='orbs.molden', skip_localization=False):

    np.set_printoptions(suppress=True)

    print("Doing Bloch Hamiltonian analysis...")

    # Put input vector into block form (only need CAS-1SF block!!)
    n_SF = wfn.n_SF
    ras1 = wfn.ras1
    ras2 = wfn.ras2
    e = wfn.e.copy()
    v_b1 = wfn.vecs[:(ras2*ras2), :].copy()
    n_roots = v_b1.shape[1]
    v_b1 = np.reshape(v_b1, (ras2,ras2,n_roots)) # v[i,a]

    # Obtain info for orbital localization and localize v
    psi4_wfn = wfn.wfn
    C = psi4.core.Matrix.to_array(psi4_wfn.Ca(), copy=True)
    if(not skip_localization):
        ras1_C = C[:, :ras1]
        ras2_C = C[:, ras1:ras1+ras2]
        ras3_C = C[:, ras1+ras2:]
        loc = psi4.core.Localizer.build('BOYS', psi4_wfn.basisset(), psi4.core.Matrix.from_array(ras2_C))
        loc.localize()
        U = psi4.core.Matrix.to_array(loc.U, copy=True)
        # localize
        v_b1 = np.einsum("ji,jbn->ibn", U, v_b1)
        v_b1 = np.einsum("ba,ibn->ian", U, v_b1)
        wfn.local_vecs = np.reshape(v_b1, (ras2*ras2, n_roots))

        C_full_loc = psi4.core.Matrix.from_array(np.column_stack((ras1_C, psi4.core.Matrix.to_array(loc.L), ras3_C)))
        # write localized orbitals to wfn and molden
        psi4_wfn.Ca().copy(C_full_loc)
        psi4_wfn.Cb().copy(C_full_loc)
        psi4.molden(psi4_wfn, molden_file)

    # determine which orbitals belong to which centers
    # C given in C_iu basis
    N = np.zeros((len(site_list), ras2))
    bas = psi4_wfn.basisset()
    S = psi4.core.Matrix.to_array(psi4_wfn.S())
    C = psi4.core.Matrix.to_array(psi4_wfn.Ca())
    CS = np.einsum("vi,vu->ui", C, S)
    for atom, A in enumerate(site_list):
        for i in range(ras2):
            for mu in range(C.shape[1]):
                if(bas.function_to_center(mu) == A):
                    print(A, mu)
                    N[atom, i] += C[mu, ras1+i] * CS[mu, ras1+i]
                    #N[atom, i] += C[ras1+i, mu] * CS[ras1+i, mu]
    print("N") # for debugging
    print(N) # for debugging

    # Obtain S
    # not needed -- S should be I if states are orthonormal

    # Remove ionic determinants

    # Extract i=a part (neutral determinants only!!)
    v_n = None
    for i in range(v_b1.shape[2]):
        v_new = np.diagonal(v_b1[:, :, i])
        if(type(v_n) == type(None)):
            v_n = v_new
        else:
            v_n = np.vstack((v_n, v_new))
    v_n = v_n.T # make sure columns are states rather than rows

    # orthonormalize (SVD)
    #v_orth = LIN.orth(v_n)
    v_orth = lowdin_orth(v_n)

    # Permute v_n appropriately
    # for each orbital, determine its center
    for i in range(ras2):
        diff = abs(N[:, i]-1)
        print(diff)
        print(np.argmin(diff))

    # Build Bloch Hamiltonian
    #H = np.dot(S, v_orth)
    H = v_orth
    H = np.dot(H, np.diag(e))
    H = np.dot(H, v_orth.T) # invert v_orth
    J = np.zeros(H.shape)
    print("Effective Hamiltonian")
    print(H)
    print("J Couplings:")
    for i in range(n_roots):
        for j in range(i):
            J[i,j] = J[j,i] = -1.0*H[i,j]
            print("\tJ%i%i = %6.6f" %(i, j, J[i,j]))

    return J


