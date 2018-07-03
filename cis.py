import numpy as np
from numpy import linalg as LIN
import psi4


def kdel(i, j):
    if(i==j):
        return 1
    else:
        return 0

def binomial(n, k):
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

def get_cis_H(wfn):
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
    # GS -> excited state
    # <0|H|ai> = h_ia + sum(<ik||ak>)
    #for detindex, det in enumerate(dets):
    #    i = det[0]
    #    a = det[1]
    #    H[0, detindex] = F[i, a]
    # <ai|H|bj>
    for d1index, det1 in enumerate(dets):
        for d2index, det2 in enumerate(dets):
            i = det1[0]
            a = det1[1]
            j = det2[0]
            b = det2[1]
            # ground state
            #if(d1index==0):
            #    H[d1index, d2index] = F[j, b]
            #elif(d2index==0):
            #    H[d1index, d2index] = F[i, a]
            # excited states
            #else:
            H[d1index, d2index] = F[a, b]*kdel(i,j) - F[i,j]*kdel(a,b) + 2.0*tei[a, j, i, b] - tei[a, j, b, i]
            # if i=j and a!=b...
            #else if(i==j and a!=b):
            #    H[det1, det2] = F[a, b] - tei[a, i, b, i]
            #else if(i!=j and a==b):
            #    H[det1, det2] = -F[i, j] - tei[i, a, j, a]
            #else if(i==j and a==b):
            #    H[det1, det2] = H[0, 0] - F[i, i] + F[a, a] - tei[i, a, i, a]

            # diagonalize H (Davidson)
    return (H, F, tei)

def cis_energy(c, H, F, tei, wfn):
        # evaluate energy
        n_dets = H.shape[0]
        # first term
        term1 = 0
        term2 = 0
        term3 = 0
        term4 = 0
        occ = wfn.doccpi()[0]
        virt = wfn.basisset().nbf() - occ 
        dets = generate_dets(occ, virt)
        n_dets = occ*virt
        for i in range(occ):
            for a in range(virt):
                term1 = term1 + c[0]*c[i*virt+a]*F[i,occ+a]
        for i in range(occ):
	    for a in range(virt):
                for b in range(virt):
                    term2 = term2 + c[i*virt+a]*c[i*virt+b]*F[occ+a,occ+b]
        for i in range(occ):
            for j in range(occ):
                for a in range(virt):
                    term3 = term3 + c[i*virt+a]*c[j*virt+a]*F[i,j]
        for i in range(occ):
            for j in range(occ):
	        for a in range(virt):
                    for b in range(virt):
                        term4 = term4 + c[i*virt+a]*c[j*virt+b]*(2.0*tei[a,j,i,b] - tei[a,j,b,i])
        E_cis = 2.0*term1 + term2 - term3 + term4
        return E_cis
            
def run():
    psi4.core.clean()
    mol = psi4.geometry("""
        0 1
        O
        H 1 1.0
        H 1 1.0 2 104.5
        symmetry c1
        """)
    psi4.set_options({'scf_type': 'pk'})
    e, wfn = psi4.energy('scf/sto-3g', molecule=mol, return_wfn=True)
    occ = wfn.doccpi()[0]
    virt = wfn.basisset().nbf() - occ
    psi4.set_options({'print': 5, 'diag_method': 'rsp', 'num_roots': 10})
    energy_cis = psi4.energy('ci1/sto-3g', molecule=mol)
    psi4.core.print_variables()
    H, F, tei = get_cis_H(wfn)
    print(H)
    print(np.sort(27.2114*LIN.eigvals(H)))
    print("REF (PSI4): ", energy_cis)
    print("FROM DIAG: ", e + np.sort(LIN.eigvals(H))[0])
    vals, vects = LIN.eig(H)
    #print("FROM CIS FN: ")
    #for i in range(vals.size):
    #    print(e + cis_energy(vects[:,i], H, F, tei, wfn))
    



