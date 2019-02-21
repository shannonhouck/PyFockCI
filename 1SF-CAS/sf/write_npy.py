import psi4
import numpy as np

# Writes Psi4 Wavefunction object to NumPy array
def wfn_to_npy(wfn, df, file='ref_wfn_mat.npz', separate=False):
    # Get MO coeffs.
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    # Get Fock matrices
    Fa = np.dot(Ca.T, np.dot(wfn.Fa(), Ca))
    Fb = np.dot(Ca.T, np.dot(wfn.Fb(), Ca))
    # get additional data
    e = wfn.energy()
    ras1 = wfn.doccpi()[0]
    ras2 = wfn.soccpi()[0]
    ras3 = wfn.basisset().nbf() - (wfn.doccpi()[0] + wfn.soccpi()[0])
    extras = np.array([e, ras1, ras2, ras3])
    # Get two-electron integrals
    # for DF case
    if(df):
        # get necessary integrals/matrices from Psi4 (AO basis)
        # get info from Psi4
        basis = wfn.basisset()
        aux = wfn.get_basisset("DF_BASIS_SCF")
        zero = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(basis)
        # (Q|pq)
        eri = psi4.core.Matrix.to_array(mints.ao_eri(zero, aux, basis, basis))
        eri = np.squeeze(eri)
        # set up J^-1/2 (don't need to keep)
        J = mints.ao_eri(zero, aux, zero, aux)
        J.power(-0.5, 1e-14)
        J = np.squeeze(J)
        if(separate):
            np.save('Ca.npy', Ca)
            np.save('Cb.npy', Cb)
            np.save('Fa.npy', Fa)
            np.save('Fb.npy', Fb)
            np.save('TEI.npy', eri)
            np.save('J.npy', J)
            np.save('extras.py', extras)
        else:
            np.savez(file, Ca=Ca, Cb=Cb, Fa=Fa, Fb=Fb, TEI=eri, J=J, extras=extras)
    # for full ERI case
    else:
        # get necessary integrals/matrices from Psi4 (AO basis)
        mints = psi4.core.MintsHelper(wfn.basisset())
        eri = psi4.core.Matrix.to_array(mints.ao_eri())
        # put in physicists' notation
        eri = eri.transpose(0, 2, 1, 3)
        # move to MO basis
        eri = np.einsum('pqrs,pa', eri, wfn.Ca())
        eri = np.einsum('aqrs,qb', eri, wfn.Ca())
        eri = np.einsum('abrs,rc', eri, wfn.Ca())
        eri = np.einsum('abcs,sd', eri, wfn.Ca())
        if(separate):
            np.save('Ca.npy', Ca)
            np.save('Cb.npy', Cb)
            np.save('Fa.npy', Fa)
            np.save('Fb.npy', Fb)
            np.save('TEI.npy', eri)
            np.save('extras.py', extras)
        else:
            np.savez(file, Ca=Ca, Cb=Cb, Fa=Fa, Fb=Fb, TEI=eri, extras=extras)

