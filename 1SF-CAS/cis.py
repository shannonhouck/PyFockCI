from __future__ import print_function
import numpy as np
from numpy import linalg as LIN
from scipy.sparse import linalg as SPLIN
from linop import LinOpH
import psi4

'''
Refs:
Crawford Tutorials (http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12)
DePrince Tutorials (https://www.chem.fsu.edu/~deprince/programming_projects/cis/)
Sherrill Notes (http://vergil.chemistry.gatech.edu/notes/cis/cis.html)
'''

# Kronaker delta function.
def kdel(i, j):
    if(i==j):
        return 1
    else:
        return 0

# Generates the set of singles determinants (spin adapted).
# First row is i, second is a.
def generate_dets_sa(n_occ, n_virt):
    n_dets = n_occ * n_virt
    det_list = np.zeros((n_dets, 2)).astype(int)
    for i in range(n_occ):
        for a in range(n_virt):
            det_list[i*n_virt+a] = [i,n_occ+a]
    return det_list

# Generates the set of singly-excited determinants (spin unadapted).
# First row is orbital excited from, second row is the orbital excited into.
# Indexing treats alphas as the first "block" and betas as the second,
# regardless of which orbitals are virtual/occupied.
# Ground state not included.
def generate_dets(na_occ, na_virt, nb_occ, nb_virt):
    nbf = na_occ + na_virt
    n_dets = (na_occ + nb_occ) * (na_virt + nb_virt)
    det_list = np.zeros((n_dets, 2)).astype(int)
    # fill out alpha -> ??
    for i in range(na_occ):
        # a -> a
        for a in range(na_virt):
            det_list[i*na_virt+a] = [i,na_occ+a]
        # a -> b
        for a in range(nb_virt):
            det_list[i*nb_virt+a+(na_virt*na_occ)] = [i,nb_occ+a+nbf]
    # fill out beta -> ??
    for i in range(nb_occ):
        # b -> a
        for a in range(na_virt):
            det_list[i*na_virt+a+(na_occ*(na_virt+nb_virt))] = [nbf+i,na_occ+a]
        # b -> b
        for a in range(nb_virt):
            det_list[i*nb_virt+a+(na_occ*(na_virt+nb_virt))+(nb_occ*na_virt)] = [i+nbf,nb_occ+a+nbf]
    return det_list

# Generates the set of singly-excited SF determinants (spin unadapted, obviously).
def generate_sf_dets(na_occ, na_virt, nb_occ, nb_virt):
    nbf = na_occ + na_virt
    n_dets = (na_occ - nb_occ) * (na_occ - nb_occ)
    det_list = np.zeros((n_dets, 2)).astype(int)
    for i in range(na_occ - nb_occ):
        for a in range(na_occ - nb_occ):
            det_list[i*(na_occ - nb_occ)+a] = [nb_occ+i, nb_occ+a+nbf]
    return det_list

# Forms the CIS Hamiltonian (not spin adapted)
def get_sf_H(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    # References: Psi4NumPy tutorials
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    C = np.block([[Ca, np.zeros_like(Cb)],
                      [np.zeros_like(Ca), Cb]])
    # one-electron part
    Fa = psi4.core.Matrix.to_array(wfn.Fa())
    Fb = psi4.core.Matrix.to_array(wfn.Fb())
    F = np.block([[Fa, np.zeros_like(Fa)],
                  [np.zeros_like(Fb), Fb]])
    F = np.dot(C.T, np.dot(F, C))
    # two-electron part
    mints = psi4.core.MintsHelper(wfn.basisset())
    tei = psi4.core.Matrix.to_array(mints.ao_eri())
    I = np.eye(2)
    tei = np.kron(I, tei)
    tei = np.kron(I, tei.T)
    tei = tei.transpose(0, 2, 1, 3) - tei.transpose(0, 2, 3, 1)
    tei = np.einsum('pqrs,pa',tei,C)
    tei = np.einsum('aqrs,qb',tei,C)
    tei = np.einsum('abrs,rc',tei,C)
    tei = np.einsum('abcs,sd',tei,C)
    # Get determinant list (hole-particle format)
    a_occ = wfn.doccpi()[0] + wfn.soccpi()[0]
    b_occ = wfn.doccpi()[0]
    a_virt = wfn.basisset().nbf() - a_occ
    b_virt = wfn.basisset().nbf() - b_occ
    na_dets = a_occ*a_virt
    nb_dets = b_occ*b_virt
    nbf = wfn.basisset().nbf()
    dets = generate_sf_dets(a_occ, a_virt, b_occ, b_virt)
    n_dets = dets.shape[0]
    # Build CIS Hamiltonian matrix
    H = np.zeros((n_dets, n_dets))
    for d1index, det1 in enumerate(dets):
        for d2index, det2 in enumerate(dets):
            i = det1[0]
            a = det1[1]
            j = det2[0]
            b = det2[1]
            if(d1index == 0 and d2index == 0):
                H[0, 0] = 0
            elif(d1index == 0):
                H[0, d2index] = 0*F[j, b]
            elif(d2index == 0):
                H[d1index, 0] = 0*F[i, a]
            else:
                H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j, i, b]
    return (H, dets, F, tei)

def do_sf_cas( charge, mult, mol, conf_space="", add_opts={}, sf_diag_method="LinOp" ):
    psi4.core.clean()
    opts = {'scf_type': 'direct',
            'basis': 'cc-pvdz',
            'reference': 'rohf',
            'guess': 'sad',
            'maxiter': 1000,
            'ci_maxiter': 50, 
            'mixed': False}
    opts.update(add_opts)
    psi4.set_options(opts)
    e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    H, dets, F, tei = get_sf_H(wfn)
    np.set_printoptions(precision=8, suppress=True)
    if(sf_diag_method == "RSP"):
        print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0:8])
        print("FROM DIAG: ", np.sort(LIN.eigvalsh(H))[0:6])
        return(e + np.sort(LIN.eigvalsh(H))[0])
    if(sf_diag_method == "LinOp"):
        #A = SPLIN.LinearOperator(H.shape, matvec=mv)
        A = LinOpH(H.shape, dets, F, tei)
        vals, vects = SPLIN.eigsh(A, which='SA')
        print("FROM LinOp:  ", vals)
        return(e + vals[0])

