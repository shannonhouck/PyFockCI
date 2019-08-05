from __future__ import print_function
import numpy as np
import psi4
import scipy.linalg as LIN

def do_bloch(v_in, e_in, n_SF, ras1, ras2, wfn):

    print("Doing Bloch Hamiltonian analysis...")

    # 1. 
    # Put input vector into block form (only need CAS-1SF block!!)

    v_b1 = v_in[:(ras2*ras2), :]
    v_b1 = np.reshape(v_b1, (ras2,ras2,v_b1.shape[1])) # v[i,a]

    # 2.
    # Obtain info for orbital localization and localize v

    # C is given in the form C_ui where i=MO and u=AO basis
    C = psi4.core.Matrix.to_array(wfn.Ca(), copy=True)
    ras2_C = C[:, ras1:ras1+ras2]
    loc = psi4.core.Localizer.build('BOYS', wfn.basisset(), psi4.core.Matrix.from_array(ras2_C))
    loc.localize()
    # U seems to be given in the form U_iu where i=MO and u=AO basis
    U = psi4.core.Matrix.to_array(loc.U, copy=True)
    # localize
    # factor of 2??
    #v_b1 = np.einsum("ij,jbn->ibn", U, v_b1)
    #v_b1 = np.einsum("ab,ibn->ian", U, v_b1)
    v_b1 = np.einsum("ji,jbn->ibn", U, v_b1)
    v_b1 = np.einsum("ba,ibn->ian", U, v_b1)

    # write localized orbitals to molden if needed
    wfn.Ca().copy(loc.L)
    wfn.Cb().copy(loc.L)
    psi4.molden(wfn, 'orbs.molden')

    # 3.
    # Obtain S

    # not needed -- S should be I if states are orthonormal

    # 4. 
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
    v_orth = LIN.orth(v_n)

    # 5.
    # Build Bloch Hamiltonian
    #H = np.dot(S, v_orth)
    H = v_orth
    H = np.dot(H, np.diag(e_in))
    H = np.dot(H, LIN.inv(v_orth)) # invert v_orth
    print("Bloch Effective Hamiltonian:")
    print(H)

    return H    


