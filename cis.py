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
# Determinant at index 0 is the ground state.
def generate_dets(na_occ, na_virt, nb_occ, nb_virt):
    nbf = na_occ + na_virt
    n_dets = (na_occ + nb_occ) * (na_virt + nb_virt)
    det_list = np.zeros((n_dets+1, 2)).astype(int)
    # fill out alpha -> ??
    for i in range(na_occ):
        # a -> a
        for a in range(na_virt):
            det_list[i*na_virt+a+1] = [i,na_occ+a]
        # a -> b
        for a in range(nb_virt):
            det_list[i*nb_virt+a+(na_virt*na_occ)+1] = [i,nb_occ+a+nbf]
    # fill out beta -> ??
    for i in range(nb_occ):
        # b -> a
        for a in range(na_virt):
            det_list[i*na_virt+a+(na_occ*(na_virt+nb_virt))+1] = [nbf+i,na_occ+a]
        # b -> b
        for a in range(nb_virt):
            det_list[i*nb_virt+a+(na_occ*(na_virt+nb_virt))+(nb_occ*na_virt)+1] = [i+nbf,nb_occ+a+nbf]
    return det_list

# Forms the CIS Hamiltonian (spin adapted)
# Could use a bit of clean-up at some point, probably.
def get_cis_H_sa(wfn):
    # get necessary integrals/matrices from Psi4
    # this gets it in AO basis, transform to MO needed
    h = psi4.core.Matrix.to_array(wfn.H()) 
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    h = np.dot(Ca.T, np.dot(h, Ca))
    F_ref = psi4.core.Matrix.to_array(wfn.Fa())
    F_ref = np.dot(Ca.T, np.dot(F_ref, Ca))
    # All of these are in AO basis...
    mints = psi4.core.MintsHelper(wfn.basisset())
    T_p4 = mints.ao_kinetic()
    T_p4.transform(wfn.Ca())
    V_p4 = mints.ao_potential()
    V_p4.transform(wfn.Ca())
    T = psi4.core.Matrix.to_array(T_p4)
    V = psi4.core.Matrix.to_array(V_p4)
    tei_ao = psi4.core.Matrix.to_array(mints.ao_eri())
    tei_mo_temp = np.einsum('pqrs,pa',tei_ao,Ca)
    tei_mo_temp = np.einsum('aqrs,qb',tei_mo_temp,Ca)
    tei_mo_temp = np.einsum('abrs,rc',tei_mo_temp,Ca)
    tei_mo_temp = np.einsum('abcs,sd',tei_mo_temp,Ca)
    tei = psi4.core.Matrix.to_array(mints.mo_eri(wfn.Ca(), wfn.Ca(), wfn.Ca(), wfn.Ca()))
    tei = np.swapaxes(tei, 1, 2)
    # determine number of dets
    occ = wfn.doccpi()[0]
    virt = wfn.basisset().nbf() - occ
    dets = generate_dets(occ, virt)
    n_dets = occ*virt
    # form Fock matrix (MO basis)
    F = np.zeros((occ+virt,occ+virt))
    for p in range(occ+virt):
        for q in range(occ+virt):
            F[p, q] = h[p, q]
            for k in range(occ):
                F[p, q] = F[p, q] + 2.0*(tei[p, k, q, k]) - tei[p, k, k, q]
    # Build CIS Hamiltonian matrix
    H = np.zeros((n_dets, n_dets))
    # <ai|H|bj>
    for d1index, det1 in enumerate(dets):
        for d2index, det2 in enumerate(dets):
            i = det1[0]
            a = det1[1]
            j = det2[0]
            b = det2[1]
            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2.0*tei[a, j, i, b] - tei[a, j, b, i]
    return (H, F, tei)

# Forms the CIS Hamiltonian (not spin adapted)
def get_cis_H(wfn):
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
    dets = generate_dets(a_occ, a_virt, b_occ, b_virt)
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

# using Davidson to solve the Hamiltonian
'''
def davidson( A, n_roots=2, e_conv=1e-6, maxIter=300 ):
    # generate vSpace (search subspace)
    # based on vInit (input vectors)
    # guess based on smallest diagonal elements in A
    guessindexes = np.diag(A).argsort()
    vInit = np.zeros((A.shape[0], n_roots))
    for i in range(n_roots):
        vInit[guessindexes[i], i] = 1.0
    vSpace = vInit;
    # size at which to collapse search subspace
    collapseSize = n_roots*A.shape[0];
    # cutoff for adding vector to Krylov search subspace
    delta = 1e-20;
    # iterations completed
    j = 0;
    sig = None;
    # index of last sigma added
    lastSig = 0;
    converged = n_roots*[False]
    finalVals = n_roots*[0]
    finalVects = n_roots*[0]

    while ( j<maxIter ):
        # form k sigma vectors
        for i in range(n_roots):
            if(not converged[i]):
                if(type(sig)==type(None)):
                    sig = np.dot(A, vSpace[:,lastSig+i])
                else:
                    sig = np.column_stack((sig, np.dot(A, vSpace[:,lastSig+i])))
            else:
                if(type(sig)==type(None)):
                    sig = np.dot(A, vSpace[:,lastSig+i])
                else:
                    sig = np.column_stack((sig, np.dot(A, vSpace[:,lastSig+i])))
  
        # form subspace matrix
        Av = np.dot(vSpace.T, sig)
        # solve for k lowest eigenvalues
        eVals, eVects = LIN.eig(Av)
        # eIndex tracks where the lowest values are
        # so we can extract the appropriate eigenvectors
        # discard high values
        eIndex = eVals.argsort()[0:n_roots]
        eVals = np.sort(eVals)[0:n_roots]

        r = None;
        # compute residuals
        for i in range(n_roots):
            if(not converged[i]):
                rNew = np.dot(sig, eVects[:,eIndex[i]]) - eVals[i]*np.dot(vSpace, eVects[:,eIndex[i]])
                if(type(r)==type(None)):
                    r = rNew
                else:
                    r = np.column_stack((r, rNew))
            else:
                rNew = np.dot(sig, eVects[:,eIndex[i]]) - eVals[i]*np.dot(vSpace, eVects[:,eIndex[i]])
                if(type(r)==type(None)):
                    r = rNew
                else:
                    r = np.column_stack((r, rNew))

        print("Iteration: %4i" % j, end='')
        for i in eVals:
            print("%16.8f" % i, end='')
        for i in range(n_roots):
            print("%16.8f" % LIN.norm(r[:,i]), end='')
        print("")

        # check residuals for convergence (break)
        for i in range(n_roots):
            if(not converged[i]):
                if( LIN.norm(r[:,i])<e_conv ):
                    converged[i] = True
                    finalVals[i] = eVals[i]
                    finalVects[i] = np.hstack((eVects[:, i], np.zeros(A.shape[0]-eVects.shape[0])))

        if all(converged):
            print("Done! \n")
            return finalVals;
        #collapse subspace if necessary (if)
        # I DON'T know for sure if this is working yet!!!
        elif( vSpace.shape[1] > collapseSize ):
            print("Y I K E S")
            lastSig = 0
            vSpaceNew = None
            for l in range(n_roots):
                newVect = np.zeros((rows(A),1))
                for i in range(vSpace.shape[1]):
                    newVect = newVect + np.dot(eVects[i,eIndex(l)], vSpace[:,i])
            # orthonormalize
            newVect = newVect/norm(newVect);
            vSpaceNew = [vSpaceNew, newVect];
            vSpace = vSpaceNew
            sig = None

        # else, apply preconditioner to residuals (elif)
        else:
            # apply preconditioner
            D = np.diag(np.diag(A))
            for i in range(n_roots):
                if(not converged[i]):
                    # construct inv(D-eVal) (preconditioner)
                    precond = D - eVals[i]*np.eye(D.shape[0])
                    for l in range(D.shape[0]):
                        if(precond[l,l] > 1e-20):
                            precond[l, l] = 1.0/precond[l,l]
                    sNew, tmp1,tmp2,tmp3 = LIN.lstsq(precond, r[:,i])
                    if ( LIN.norm(sNew) > delta ):
                        # orthogonalize
                        h = np.dot(vSpace.T, sNew);
                        sNew = sNew - np.dot(vSpace,h);
                        # normalize
                        sNew = (1.0/LIN.norm(sNew))*sNew
                        vSpace = np.column_stack((vSpace, sNew))
                        lastSig = lastSig + 1;
                    else:
                        vSpace = np.column_stack((vSpace, vSpace[:,0]))
                        lastSig = lastSig+1
                else:
                    vSpace = np.column_stack((vSpace, finalVects[i]))
                    lastSig = lastSig + 1
        if ( vSpace.shape[1] > A.shape[1] ):
            print("...\nError: Make sure your inputs are reasonable!\n\n");
            return 0;
        # increase j and loop again
        j = j+1;
'''

def run():
    psi4.core.clean()
    mol = psi4.geometry("""
        0 3
        N 0 0 0
        N 1.5 0 0
        symmetry c1
        """)
    '''
    mol = psi4.geometry("""
        0 3
        O   0.000000000000  -0.143225816552   0.000000000000
        H   1.638036840407   1.136548822547  -0.000000000000
        H  -1.638036840407   1.136548822547  -0.000000000000
        symmetry c1
        """)
    '''
    #psi4.set_options({'scf_type': 'direct', 'reference': 'rhf', 'e_convergence': 1e-10, 'd_convergence': 1e-10})
    #psi4.set_options({'scf_type': 'direct', 'reference': 'uhf', 'e_convergence': 1e-10, 'd_convergence': 1e-10})
    psi4.set_options({'scf_type': 'pk', 'reference': 'uhf', 'e_convergence': 1e-8, 'd_convergence': 1e-8})
    e, wfn = psi4.energy('scf/sto-3g', molecule=mol, return_wfn=True)
    occ = wfn.doccpi()[0]
    virt = wfn.basisset().nbf() - occ
    #psi4.set_options({'diag_method': 'rsp', 'e_convergence': 1e-10, 'r_convergence': 1e-10, 'ex_level': 1})
    #energy_cis2 = psi4.energy('ci1/sto-3g', molecule=mol, ref_wfn=wfn)
    energy_cis2 = 0
    #psi4.core.clean()
    #psi4.set_options({'num_roots': 12, 'diag_method': 'rsp', 'e_convergence': 1e-10, 'r_convergence': 1e-10, 'ras1': [5], 'ras2': [4], 'ras3': [1], 'ex_level': 1})
    #energy_cis1 = psi4.energy('detci/sto-3g', molecule=mol, ref_wfn=wfn)
    #psi4.core.print_variables()
    H, dets, F, tei = get_cis_H(wfn)
    #Hr, Fr, teir = get_cis_H_rhf(wfn)
    #print(e*np.eye(Hr.shape[0]) + Hr)
    print("REF", energy_cis2)
    #print("REF (PSI4): ", energy_cis1 - e)
    #print("REF (PSI4): ", energy_cis2 - e)
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 1 TOTAL ENERGY'))
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 2 TOTAL ENERGY'))
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 3 TOTAL ENERGY'))
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 4 TOTAL ENERGY'))
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 5 TOTAL ENERGY'))
    #print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 6 TOTAL ENERGY'))
    #print(H + e*np.eye(H.shape[0]))
    #print(H[1:, 1:])
    #vals = davidson(H[1:, 1:], n_roots=6, e_conv = 1e-5)
    np.set_printoptions(precision=8, suppress=True)
    
    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0:10])
    print("FROM DIAG: ", np.sort(LIN.eigvalsh(H))[0:10])

    #A = SPLIN.LinearOperator(H[1:, 1:].shape, matvec=mv)
    A = LinOpH(H.shape, dets, F, tei)
    vals, vects = SPLIN.eigsh(A, which='SA')
    print("FROM DAV:  ", vals)
    #print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(Hr))[0])
    #vals, vects = LIN.eig(H)
    #print("FROM CIS FN: ")
    #for i in range(vals.size):
    #    print(e + cis_energy(vects[:,i], H, F, tei, wfn))
    



