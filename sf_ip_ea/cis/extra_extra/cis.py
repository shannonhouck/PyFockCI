import numpy as np
from numpy import linalg as LIN
import psi4

# Kronaker delta function.
def kdel(i, j):
    if(i==j):
        return 1
    else:
        return 0

# Generates the set of singles determinants.
# First row is i, second is a.
def generate_dets(n_occ, n_virt):
    n_dets = n_occ * n_virt
    det_list = np.zeros((n_dets, 2)).astype(int)
    for i in range(n_occ):
        for a in range(n_virt):
            det_list[i*n_virt+a] = [i,n_occ+a]
    return det_list

def generate_dets_ab(na_occ, na_virt, nb_occ, nb_virt):
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
            det_list[i*na_virt+a+(na_virt*na_occ)] = [i,nb_occ+a+nbf]
    # fill out beta -> ??
    for i in range(nb_occ):
        # b -> a
        for a in range(na_virt):
            det_list[i*na_virt+a+(na_occ*(na_virt+nb_virt))] = [nbf+i,na_occ+a]
        # b -> b
        for a in range(nb_virt):
            det_list[i*nb_virt+a+(na_occ*(na_virt+nb_virt))+(nb_occ*na_virt)] = [i+nbf,nb_occ+a+nbf]
    return det_list

# Forms the CIS Hamiltonian
def get_cis_H_rhf(wfn):
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
    print(np.allclose(T+V, h))
    tei_ao = psi4.core.Matrix.to_array(mints.ao_eri())
    tei_mo_temp = np.einsum('pqrs,pa',tei_ao,Ca)
    tei_mo_temp = np.einsum('aqrs,qb',tei_mo_temp,Ca)
    tei_mo_temp = np.einsum('abrs,rc',tei_mo_temp,Ca)
    tei_mo_temp = np.einsum('abcs,sd',tei_mo_temp,Ca)
    tei = psi4.core.Matrix.to_array(mints.mo_eri(wfn.Ca(), wfn.Ca(), wfn.Ca(), wfn.Ca()))
    print(np.allclose(tei, tei_mo_temp))
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
    np.set_printoptions(suppress=True,precision=4,linewidth=800)
    # checking F
    print(np.allclose(F, F_ref))
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

# Forms the CIS Hamiltonian
def get_cis_H(wfn):
    np.set_printoptions(suppress=True,precision=4,linewidth=800)
    # get necessary integrals/matrices from Psi4
    # this gets it in AO basis, transform to MO needed
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    h = psi4.core.Matrix.to_array(wfn.H())
    h = np.dot(Ca.T, np.dot(h, Ca))
    Fa = psi4.core.Matrix.to_array(wfn.Fa())
    Fb = psi4.core.Matrix.to_array(wfn.Fb())
    Fa = np.dot(Ca.T, np.dot(Fa, Ca))
    Fb = np.dot(Cb.T, np.dot(Fb, Cb))
    # All of these are in AO basis...
    mints = psi4.core.MintsHelper(wfn.basisset())
    tei_ao = psi4.core.Matrix.to_array(mints.ao_eri())
    tei_mo_temp = np.einsum('pqrs,pa',tei_ao,Ca)
    tei_mo_temp = np.einsum('aqrs,qb',tei_mo_temp,Cb)
    tei_mo_temp = np.einsum('abrs,rc',tei_mo_temp,Cb)
    tei_mo_temp = np.einsum('abcs,sd',tei_mo_temp,Ca)
    tei = psi4.core.Matrix.to_array(mints.mo_eri(wfn.Ca(), wfn.Ca(), wfn.Ca(), wfn.Ca()))
    print(np.allclose(tei, tei_mo_temp))
    tei = np.swapaxes(tei, 1, 2)
    # determine number of dets
    a_occ = wfn.doccpi()[0] + wfn.soccpi()[0]
    b_occ = wfn.doccpi()[0]
    a_virt = wfn.basisset().nbf() - a_occ
    b_virt = wfn.basisset().nbf() - b_occ
    na_dets = a_occ*a_virt
    nb_dets = b_occ*b_virt
    nbf = wfn.basisset().nbf()
    dets = generate_dets_ab(a_occ, a_virt, b_occ, b_virt)
    n_dets = dets.shape[0]
    np.set_printoptions(suppress=True,precision=4,linewidth=800)
    # Build CIS Hamiltonian matrix
    H = np.zeros((n_dets, n_dets))
    # <ai|H|bj>
    F = np.block([[Fa, np.zeros((Fa.shape[0],Fb.shape[1]))],
                  [np.zeros((Fb.shape[0],Fa.shape[1])), Fb]])
    print(F)
    fill = 0*tei
    print(tei.shape)
    tei = np.block([[[[tei, fill], [fill, fill]], [[fill,tei],[fill,fill]]], [[[fill,fill],[tei,fill]], [[fill,fill],[fill, tei]]]])
    #print(tei.shape)
    #print(tei.shape)
    #print(tei[0,1,1,0]) # all alpha, not zero
    #print(tei[1,1,11,11]) # all alpha all beta, not zero
    #print(tei[11,1,11,1]) # all alpha all beta, not zero
    #print(tei[10,11,11,10]) # all beta, not zero
    #print(tei[0,12,12,0]) # alpha and beta, zero
    #print(tei[12,0,0,12]) # alpha and beta, zero
    #print(tei[12,0,12,0]) # alpha and beta, not zero
    #print(tei[0,12,0,12]) # alpha and beta, not zero
    for d1index, det1 in enumerate(dets):
        for d2index, det2 in enumerate(dets):
            i = det1[0]
            a = det1[1]
            j = det2[0]
            b = det2[1]
            '''
            # if i in alpha
            if(i < nbf):
                # if a in alpha
                if(a < nbf):
                    if(j < nbf):
                        # all in alpha
                        if(b < nbf):
                            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2*tei[a, j, i, b] - tei[a, j, b, i]
                        # i,a,j alpha, b beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j, i, b-nbf]
                    else:
                        # i,a,b in alpha, j in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j-nbf, i, b]
                        # i,a in alpha, j,b, in beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j-nbf, i, b-nbf] - tei[a, j-nbf, b-nbf, i]
                else:
                    if(j < nbf):
                        # i,j,b in alpha, a in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 # F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a-nbf, j, i, b]
                        # i,j alpha, a,b beta
                        else:
                            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2*tei[a-nbf, j, i, b-nbf] - tei[a-nbf, j, b-nbf, i]
                    else:
                        # i,b in alpha, a,j in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a-nbf, j-nbf, i, b]
                        # i in alpha, j,b,a in beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a-nbf, j-nbf, i, b-nbf]
            else:
                # if a in alpha
                if(a < nbf):
                    if(j < nbf):
                        # a,j,b in alpha, i in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j, i-nbf, b]
                        # a,j alpha, i,b beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j, i-nbf, b-nbf]
                    else:
                        # a,b in alpha, i,j in beta
                        if(b < nbf):
                            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2*tei[a, j-nbf, i-nbf, b] - tei[a, j-nbf, b, i-nbf]
                        # a in alpha, i,j,b, in beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j-nbf, i-nbf, b-nbf]
                else:
                    if(j < nbf):
                        # j,b in alpha, i,a in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) +tei[a-nbf, j, i-nbf, b] - tei[a-nbf, j, b, i-nbf]
                        # j alpha, i,a,b beta
                        else:
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a-nbf, j, i-nbf, b-nbf]
                    else:
                        # b in alpha, i,a,j in beta
                        if(b < nbf):
                            H[d1index, d2index] = 0 #F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a-nbf, j-nbf, i-nbf, b]
                        # i,j,b,a in beta
                        else:
                            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2*tei[a-nbf,j-nbf,i-nbf,b-nbf] - tei[a-nbf, j-nbf, b-nbf, i-nbf]
            '''
            # if i->a in beta set
            #H[d1index, d2index] = Fa[a, b]*kdel(i,j) - Fa[i,j]*kdel(a,b) + 2.0*tei[a, j, i, b] - tei[a, j, b, i]
            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a, j, i, b] - tei[a, j, b, i]
    return (H, Fa, tei)

def run():
    psi4.core.clean()
    mol = psi4.geometry("""
        0 3
        O   0.000000000000  -0.143225816552   0.000000000000
        H   1.638036840407   1.136548822547  -0.000000000000
        H  -1.638036840407   1.136548822547  -0.000000000000
        symmetry c1
        """)
    psi4.set_options({'scf_type': 'direct', 'e_convergence': 1e-10, 'd_convergence': 1e-10, 'reference': 'rohf'})
    e, wfn = psi4.energy('scf/sto-3g', molecule=mol, return_wfn=True)
    occ = wfn.doccpi()[0]
    virt = wfn.basisset().nbf() - occ
    psi4.set_options({'print': 5, 'diag_method': 'rsp', 'num_roots': 10, 'e_convergence': 1e-10})
    energy_cis = psi4.energy('ci1/sto-3g', molecule=mol, ref_wfn=wfn)
    psi4.core.print_variables()
    H, F, tei = get_cis_H(wfn)
    Hr, Fr, teir = get_cis_H_rhf(wfn)
    print(e*np.eye(H.shape[0]) + H)
    print(e*np.eye(Hr.shape[0]) + Hr)
    print(np.allclose(H, H.T))
    print(np.sort(27.2114*LIN.eigvalsh(H)))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 0 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 1 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 2 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 3 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 4 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 5 TOTAL ENERGY'))
    print("REF (PSI4): ", psi4.core.get_variable('CI ROOT 6 TOTAL ENERGY'))
    print(np.allclose(H, H.T))
    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0])
    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(Hr))[0])
    vals, vects = LIN.eig(H)
    #print("FROM CIS FN: ")
    #for i in range(vals.size):
    #    print(e + cis_energy(vects[:,i], H, F, tei, wfn))
    



