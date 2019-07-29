from __future__ import print_function
import numpy as np
import psi4
import scipy.linalg as LIN

def do_bloch(v_in, e_in, n_SF, ras1, ras2, wfn):

    print("Doing Bloch Hamiltonian analysis...")

    # localize orbitals needed
    C = psi4.core.Matrix.to_array(wfn.Ca(), copy=True)
    ras2_C = C[:, ras1:ras1+ras2]
    loc = psi4.core.Localizer.build('BOYS', wfn.basisset(), psi4.core.Matrix.from_array(ras2_C))
    loc.localize()
    U = psi4.core.Matrix.to_array(loc.U, copy=True)

    # Put input vector into block form (only need CAS-1SF block!!)
    v_b1 = v_in[:(ras2*ras2), :]
    v_b1 = np.reshape(v_b1, (ras2,ras2,v_b1.shape[1])) # v[i,a]

    # localize
    v_b1 = np.einsum("ij,jbn->ibn", U, v_b1)
    v_b1 = np.einsum("ba,ibn->ian", U, v_b1)

    # Extract i=a part (neutral determinants only!!)
    v_n = np.diagonal(v_b1, axis1=0, axis2=1).T

    # print these coeffs (debug)
    #print(v_n)

    # remove eigenvectors/eigenvalues close to zero
    print("Removing states which are primarily ionic...")
    cutoff = 1e-10
    n_removed = []
    v_keep = None
    for i in range(v_n.shape[1]):
        # if above cutoff, orthogonalize and add to space
        if(np.sum(np.abs(v_n[:,i])) > cutoff):
            v_new = v_n[:,i]
            e_new = e_in[i]
            if(type(v_keep) == type(None)):
                print("added a vect")
                v_keep = v_new
                e_keep = np.array([e_new])
            else:
                print("added a vect")
                # orthogonalize
                v_keep = np.vstack((v_keep, v_new))
                e_keep = np.hstack((e_keep, np.array([e_new])))
        else:
            n_removed = n_removed + [i]
    print("Removed the following states: ")
    print(n_removed)
    print(e_keep.shape)
    print(v_keep.shape)
    # orthonormalize (SVD)
    v_orth = LIN.orth(v_keep)
    print(v_orth.shape)

    # build Bloch Hamiltonian
    H = np.dot(v_orth.T, np.dot(np.diag(e_keep), v_orth))
    print("Bloch Effective Hamiltonian:")
    print(H)

    return H    


