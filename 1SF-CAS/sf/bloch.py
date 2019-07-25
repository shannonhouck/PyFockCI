from __future__ import print_function
import numpy as np
import scipy.linalg as LIN

def do_bloch(v_in, e_in, n_SF, ras2):

    print("Doing Bloch Hamiltonian analysis...")

    # Put input vector into block form (only need CAS-1SF block!!)
    v_b1 = v_in[:(ras2*ras2), :]
    v_b1 = np.reshape(v_b1, (ras2,ras2,v_b1.shape[1])) # v for block 1
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
                v_keep = v_new
                e_keep = np.array([e_new])
            else:
                # orthogonalize
                v_keep = np.vstack((v_keep, v_new))
                e_keep = np.stack((e_keep, np.array([e_new])))
        else:
            n_removed = n_removed + [i]
    print("Removed the following states: ")
    print(n_removed)
    # orthonormalize (SVD)
    v_orth = LIN.orth(v_keep.T)
    print(v_orth.shape)

    # build Bloch Hamiltonian
    H = np.dot(v_orth, np.dot(np.diag(e_keep[:, 0]), v_orth.T))
    print("Bloch Effective Hamiltonian:")
    print(H)
    


