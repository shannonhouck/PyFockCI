from __future__ import print_function
import numpy as np
from .tei import *

# Calculates S**2 for a given CI state.
# Parameters:
#    n_SF            Number of spin-flips
#    delta_ec        Change in electron count
#    conf_space      Excitation scheme (Options: "", "h", "p", "h,p")
#    vect            Eigenvector (CI coefficients)
#    docc            Number of doubly occupied orbitals
#    socc            Number of singly occupied orbitals
#    virt            Number of doubly unoccupied orbitals
# Returns:
#    s2              The S**2 expectation value for the state
def calc_s_squared(n_SF, delta_ec, conf_space, vect, docc, socc, virt):
    s2 = 0.0
    # obtain Sz and Sz (same general formula regardless of method)
    na = socc - n_SF
    nb = n_SF
    if(delta_ec==1): # IP
        nb = nb + 1
    if(delta_ec==-1): # EA
        na = na - 1
    for v in vect:
        # do Sz
        s2 = s2 + v*v*(0.5*(na) - 0.5*(nb))
        # do Sz^2
        s2 = s2 + v*v*(0.25*(na*na) + 0.25*(nb*nb) - 0.5*(na*nb))

    # CAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        vect = np.reshape(vect, (socc,socc))
        for p in range(socc):
            for q in range(socc):
                s2 = s2 + vect[p,p]*vect[q,q]*1.0
        return s2

    # CAS-2SF
    if(n_SF==2 and delta_ec==0 and conf_space==""):
        v_ref1 = np.zeros((socc, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for a in range(socc):
                    for b in range(a):
                        v_ref1[i, j, a, b] = vect[index]
                        v_ref1[i, j, b, a] = -1.0*vect[index]
                        v_ref1[j, i, a, b] = -1.0*vect[index]
                        v_ref1[j, i, b, a] = vect[index]
                        index = index + 1
        for b in range(socc):
            for j in range(socc):
                for p in range(socc):
                    for q in range(socc):
                        s2 = s2 + v_ref1[q,j,q,b]*v_ref1[p,j,p,b]*1.0
        return s2

    # 1IP and 1EA
    if(n_SF==0 and (delta_ec==-1 or delta_ec==1) and conf_space==""):
        # no contributions from S-S+
        return s2

    # RAS(h)-1IP
    if(n_SF==0 and delta_ec==-1 and conf_space=="h"):
        v_b1 = vect[0:socc] # v for block 1
        v_b2 = vect[socc:docc+socc] # v for block 2
        v_b3 = vect[docc+socc:] # v for block 3
        # v(1) indexing: (i:a)
        v_ref1 = np.reshape(v_b1, (socc))
        # v(2) indexing: (I:a)
        v_ref2 = np.reshape(v_b2, (docc))
        # v(3) indexing: (Iia:bba)
        v_ref3 = np.reshape(v_b3, (docc, socc, socc))
        # block 2
        s2 = s2 + np.einsum("I,I->", v_ref2, v_ref2)
        for p in range(docc):
            for q in range(socc):
                s2 = s2 + 2.0*v_ref2[p]*v_ref3[p,q,q]
        # block 3
        for p in range(socc):
            for q in range(socc):
                for I in range(docc):
                    s2 = s2 + v_ref3[I,p,p]*v_ref3[I,q,q]
        return s2

    # RAS(p)-IP
    if(n_SF==0 and delta_ec==-1 and conf_space=="p"):
        # no contributions from S-S+
        return s2

    # RAS(h)-EA
    if(n_SF==0 and delta_ec==1 and conf_space=="h"):
        # no contributions from S-S+
        return s2

    # RAS(p)-EA
    if(n_SF==0 and delta_ec==1 and conf_space=="p"):
        v_b1 = vect[0:socc] # v for block 1
        v_b2 = vect[socc:socc+virt] # v for block 2
        v_b3 = vect[socc+virt:] # v for block 3
        # v(1) indexing: (a:b)
        v_ref1 = np.reshape(v_b1, (socc))
        # v(2) indexing: (A:b)
        v_ref2 = np.reshape(v_b2, (virt))
        # v(3) indexing: (iAa:aab)
        v_ref3 = np.reshape(v_b3, (virt, socc, socc))
        # block 2
        s2 = s2 + np.einsum("A,A->", v_ref2, v_ref2)
        for A in range(virt):
            for p in range(socc):
                s2 = s2 - 2.0*v_ref2[A]*v_ref3[A,p,p]
        # block 3
        for A in range(virt):
            for p in range(socc):
                for q in range(socc):
                    s2 = s2 + v_ref3[A,p,p]*v_ref3[A,q,q]
        return s2

    # CAS-1SF-EA
    if(n_SF==1 and delta_ec==1 and conf_space==""):
        # v(1) unpack to indexing: (iab:abb)
        v_ref1 = np.zeros((socc,socc,socc))
        index = 0
        for i in range(socc):
            for a in range(socc):
                for b in range(a):
                    v_ref1[i, a, b] = vect[index]
                    v_ref1[i, b, a] = -1.0*vect[index]
                    index = index + 1
        for p in range(socc):
            for q in range(socc):
                for a in range(socc):
                        s2 = s2 + v_ref1[q,q,a]*v_ref1[p,p,a]
        return s2

    # CAS-1SF-IP
    if(n_SF==1 and delta_ec==-1 and conf_space==""):
        # v(1) unpack to indexing: (iab:abb)
        v_ref1 = np.zeros((socc,socc,socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for a in range(socc):
                    v_ref1[i, j, a] = vect[index]
                    v_ref1[j, i, a] = -1.0*vect[index]
                    index = index + 1
        for p in range(socc):
            for q in range(socc):
                for i in range(socc):
                        s2 = s2 + v_ref1[q,i,q]*v_ref1[p,i,p]
        return s2

    # RAS(h)-1SF-IP
    if(n_SF==1 and delta_ec==-1 and conf_space=="h"):
        n_b1_dets = int(socc * ((socc-1)*(socc)/2))
        n_b2_dets = int(socc * docc * socc)
        v_b1 = vect[0:n_b1_dets]
        v_b2 = vect[n_b1_dets:n_b1_dets+n_b2_dets]
        v_b3 = vect[n_b1_dets+n_b2_dets:]
        # v(1) unpack to indexing: (ija:aab)
        v_ref1 = np.zeros((socc,socc,socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for a in range(socc):
                    v_ref1[i, j, a] = v_b1[index]
                    v_ref1[j, i, a] = -1.0*v_b1[index]
                    index = index + 1
        # v(2) unpack to indexing: (Iia:aab)
        v_ref2 = np.reshape(v_b2, (docc, socc, socc))
        # v(3) unpack to indexing: (Iijab:aaabb)
        v_ref3 = np.zeros((docc, socc, socc, socc, socc))
        index = 0
        for I in range(docc):
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        for b in range(a):
                            v_ref3[I, i, j, a, b] = v_b3[index]
                            v_ref3[I, j, i, a, b] = -1.0*v_b3[index]
                            v_ref3[I, i, j, b, a] = -1.0*v_b3[index]
                            v_ref3[I, j, i, b, a] = v_b3[index]
                            index = index + 1
        # block 1 -- CORRECT FOR SURE
        for p in range(socc):
            for q in range(socc):
                for i in range(socc):
                    s2 = s2 + v_ref1[q,i,q]*v_ref1[p,i,p]
        # block 2
        for I in range(docc):
            for p in range(socc):
                for q in range(socc):
                    s2 = s2 + v_ref2[I,p,p]*v_ref2[I,q,q]
        for I in range(docc):
            for a in range(socc):
                for i in range(socc):
                    s2 = s2 + v_ref2[I,i,a]*v_ref2[I,i,a]
        # block 2 w/ block 3
        for I in range(docc):
            for p in range(socc):
                for q in range(socc):
                    for a in range(socc):
                        for i in range(socc):
                            s2 = s2 - 2.0*v_ref2[I,i,a]*v_ref3[I,p,i,a,p]
        # block 2
        for I in range(docc):
            for p in range(socc):
                for q in range(socc):
                    for a in range(socc):
                        for i in range(socc):
                            s2 = s2 + v_ref3[I,q,i,a,q]*v_ref3[I,p,i,a,p]
                            #s2 = s2 - v_ref3[I,i,q,q,a]*v_ref3[I,i,p,p,a]
        return s2

    # RAS(p)-1SF-EA
    if(n_SF==1 and delta_ec==1 and conf_space=="p"):
        n_b1_dets = int(socc * ((socc-1)*(socc)/2))
        n_b2_dets = int(socc * virt * socc)
        v_b1 = vect[0:n_b1_dets]
        v_b2 = vect[n_b1_dets:n_b1_dets+n_b2_dets]
        v_b3 = vect[n_b1_dets+n_b2_dets:]
        # v(1) unpack to indexing: (iab:abb)
        v_ref1 = np.zeros((socc,socc,socc))
        index = 0
        for i in range(socc):
            for a in range(socc):
                for b in range(a):
                    v_ref1[i, a, b] = v_b1[index]
                    v_ref1[i, b, a] = -1.0*v_b1[index]
                    index = index + 1
        # v(2) unpack to indexing: (Aia:abb)
        v_ref2 = np.reshape(v_b2, (virt, socc, socc))
        # v(3) unpack to indexing: (Aijab:aaabb)
        v_ref3 = np.zeros((virt, socc, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for A in range(virt):
                    for a in range(socc):
                        for b in range(a):
                            v_ref3[A, i, j, a, b] = v_b3[index]
                            v_ref3[A, j, i, a, b] = -1.0*v_b3[index]
                            v_ref3[A, i, j, b, a] = -1.0*v_b3[index]
                            v_ref3[A, j, i, b, a] = v_b3[index]
                            index = index + 1
        # block 1
        for p in range(socc):
            for q in range(socc):
                for a in range(socc):
                    s2 = s2 + v_ref1[q,q,a]*v_ref1[p,p,a]
        # block 2
        for A in range(virt):
            for p in range(socc):
                for q in range(socc):
                    s2 = s2 + v_ref2[A,p,p]*v_ref2[A,q,q]
                    for a in range(p):
                        for i in range(socc):
                            s2 = s2 + 2.0*v_ref2[A,i,a]*v_ref3[A,i,p,p,a]
        # block 2
        for A in range(virt):
            for p in range(socc):
                for q in range(socc):
                    for a in range(min(p,q)):
                        for i in range(socc):
                            s2 = s2 + v_ref3[A,i,q,q,a]*v_ref3[A,i,p,p,a]
        return s2

    # RAS(h)-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="h"):
        v_b1 = vect[:(socc*socc)] # v for block 1
        v_b2 = vect[(socc*socc):((socc*socc)+(docc*socc))] # v for block 2
        v_b3 = vect[((socc*socc)+(docc*socc)):] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (socc, socc))
        # v(2) indexing: (Ai:ab)
        v_ref2 = np.reshape(v_b2, (docc, socc))
        # v(3) unpack to indexing: (Aijb:aaab)
        v_ref3 = np.zeros((docc, socc, socc, socc))
        index = 0
        for I in range(docc):
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref3[I, i, a, b] = v_b3[index]
                        v_ref3[I, i, b, a] = -1.0*v_b3[index]
                        index = index + 1
        # block 1
        for p in range(socc):
            for q in range(socc):
                s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2
        #s2 = s2 + np.einsum("Ia,Ia->", v_ref2, v_ref2)
        for I in range(docc):
            for a in range(socc):
                s2 = s2 + v_ref2[I,a]*v_ref2[I,a]
                for p in range(socc):
                    s2 = s2 + 2.0*v_ref2[I,a]*v_ref3[I,p,a,p]
        # block 3
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    for q in range(socc):
                        s2 = s2 + v_ref3[I,p,a,p]*v_ref3[I,q,a,q]
                        s2 = s2 - v_ref3[I,p,a,p]*v_ref3[I,q,q,a]
        return s2


    # RAS(h,p)-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="h,p"):
        n_b1_dets = int(socc*socc)
        n_b2_dets = int(docc*socc)
        n_b3_dets = int(socc*virt)
        n_b4_dets = int(docc*socc*(socc*(socc-1)/2))
        n_b5_dets = int(virt*socc*(socc*(socc-1)/2))
        v_b1 = vect[0:n_b1_dets] # v for block 1
        v_b2 = vect[n_b1_dets:n_b1_dets+n_b2_dets] # v for block 2
        v_b3 = vect[n_b1_dets+n_b2_dets:n_b1_dets+n_b2_dets+n_b3_dets] # v for block 3
        v_b4 = vect[n_b1_dets+n_b2_dets+n_b3_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets] # v for block 4
        v_b5 = vect[n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets+n_b5_dets] # v for block 5
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (socc, socc))
        # v(2) indexing: (Ia:ab)
        v_ref2 = np.reshape(v_b2, (docc, socc))
        # v(3) indexing: (Ai:ab)
        v_ref3 = np.reshape(v_b3, (virt, socc))
        # v(3) unpack to indexing: (Iiab:babb)
        v_ref4 = np.zeros((docc, socc, socc, socc))
        index = 0
        for I in range(docc):
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref4[I, i, a, b] = v_b4[index]
                        v_ref4[I, i, b, a] = -1.0*v_b4[index]
                        index = index + 1
        # v(5) unpack to indexing: (Aijb:aaab)
        v_ref5 = np.zeros((virt, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for A in range(virt):
                    for b in range(socc):
                        v_ref5[A, i, j, b] = v_b5[index]
                        v_ref5[A, j, i, b] = -1.0*v_b5[index]
                        index = index + 1
        # block 1
        for p in range(socc):
            for q in range(socc):
                s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2
        s2 = s2 + np.einsum("Ia,Ia->", v_ref2, v_ref2)
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - 2.0*v_ref2[I,a]*v_ref4[I,p,a,p]
        # block 3
        s2 = s2 + np.einsum("Ai,Ai->", v_ref3, v_ref3)
        for A in range(virt):
            for i in range(socc):
                for p in range(socc):
                    s2 = s2 + 2.0*v_ref3[A,i]*v_ref5[A,i,p,p]
        # block 4
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    for q in range(socc):
                        s2 = s2 + v_ref4[I,p,a,p]*v_ref4[I,q,a,q]
        # block 5
        for A in range(virt):
            for i in range(socc):
                for q in range(socc):
                    for p in range(socc):
                        s2 = s2 + v_ref5[A,i,q,q]*v_ref5[A,i,p,p]
        return s2

    # RAS(p)-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="p"):
        v_b1 = vect[:(socc*socc)] # v for block 1
        v_b2 = vect[(socc*socc):(socc*socc)+(socc*virt)] # v for block 2
        v_b3 = vect[(socc*socc)+(socc*virt):] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (socc, socc))
        # v(2) indexing: (Ai:ab)
        v_ref2 = np.reshape(v_b2, (virt, socc))
        # v(3) unpack to indexing: (Aijb:aaab)
        v_ref3 = np.zeros((virt, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for A in range(virt):
                    for b in range(socc):
                        v_ref3[A, i, j, b] = v_b3[index]
                        v_ref3[A, j, i, b] = -1.0*v_b3[index]
                        index = index + 1
        # block 1
        for p in range(socc):
            for q in range(socc):
                s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2
        s2 = s2 + np.einsum("Ai,Ai->", v_ref2, v_ref2)
        for A in range(virt):
            for i in range(socc):
                for p in range(socc):
                    s2 = s2 + 2.0*v_ref2[A,i]*v_ref3[A,i,p,p]
        # block 3
        for A in range(virt):
            for i in range(socc):
                for q in range(socc):
                    for p in range(socc):
                        s2 = s2 + v_ref3[A,i,q,q]*v_ref3[A,i,p,p]
        return s2

    else:
        return s2 + smp_with_eri(n_SF, delta_ec, conf_space, vect, docc, socc, virt)

    #return s2 + smp_with_eri(n_SF, delta_ec, conf_space, vect, docc, socc, virt)

# Returns a list of determinants in the following form:
# det = [...] (det[0] is 0th determinant, det[1] is 1st, etc.)
# det[count] = [[[elim (alpha)], [elim (beta)]], [[add (a)], [add (b)]]]
# indexing starts at zero
# complicated? yeah, for sure, but it works
def generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3):
    # storing list of determinants
    dets_list = []

    # CAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        # make array with determinant data (i->a)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[a]]])
        return dets_list

    # RAS(h)-1SF
    elif(n_SF==1 and delta_ec==0 and conf_space=="h"):
        # v(1) indexing: (ia:ab)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[a]]])
        # v(2) indexing: (Ia:ab)
        for I in range(ras1):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[I],[]], [[],[a]]])
        # v(3) unpack to indexing: (Iiab:babb)
        for I in range(ras1):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    for b in range(ras1,a):
                        dets_list.append([[[i],[I]], [[],[a,b]]])
        return dets_list

    # RAS(p)-1SF
    elif(n_SF==1 and delta_ec==0 and conf_space=="p"):
        # v(1) indexing: (ia:ab)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[a]]])
        # v(2) indexing: (Ai:ab)
        for A in range(ras1+ras2,ras1+ras2+ras3):
            for i in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[A]]])
        # v(3) unpack to indexing: (Aijb:aaab)
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for A in range(ras1+ras2,ras1+ras2+ras3):
                    for b in range(ras1,ras1+ras2):
                        dets_list.append([[[i,j],[]], [[A],[b]]])
        return dets_list

    # RAS(h,p)-1SF
    elif(n_SF==1 and delta_ec==0 and conf_space=="h,p"):
        # v(1) indexing: (ia:ab)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[a]]])
        # v(2) indexing: (Ia:ab)
        for I in range(ras1):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[I],[]], [[],[a]]])
        # v(3) indexing: (Ai:ab)
        for A in range(ras1+ras2,ras1+ras2+ras3):
            for i in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[A]]])
        # v(4) unpack to indexing: (Iiab:babb)
        for I in range(ras1):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    for b in range(ras1,a):
                        dets_list.append([[[i],[I]], [[],[a,b]]])
        # v(5) unpack to indexing: (Aijb:aaab)
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for A in range(ras1+ras2,ras1+ras2+ras3):
                    for b in range(ras1,ras1+ras2):
                        dets_list.append([[[i,j],[]], [[A],[b]]])
        return dets_list

    # CAS-2SF
    elif(n_SF==2 and delta_ec==0 and conf_space==""):
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for a in range(ras1,ras1+ras2):
                    for b in range(ras1,a):
                        dets_list.append([[[i,j],[]], [[],[a,b]]])

    # CAS-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space==""):
        # v(1) indexing: (i:a)
        for i in range(ras1,ras1+ras2):
            dets_list.append([[[i],[]], [[],[]]])

    # RAS(h)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="h"):
        # v(1) indexing: (i:a)
        for i in range(ras1,ras1+ras2):
            dets_list.append([[[i],[]], [[],[]]])
        # v(2) indexing: (I:a)
        for I in range(ras1):
            dets_list.append([[[I],[]], [[],[]]])
        # v(3) indexing: (Iia:bab)
        for I in range(ras1):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[i],[I]], [[],[a]]])

    # RAS(p)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="p"):
        # v(1) indexing: (i:a)
        for i in range(ras1,ras1+ras2):
            dets_list.append([[[i],[]], [[],[]]])
        # v(2) indexing: (Aij:aaa)
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for A in range(ras1+ras2,ras1+ras2+ras3):
                    dets_list.append([[[i,j],[]], [[A],[]]])

    # CAS-1SF-IP
    elif(n_SF==1 and delta_ec==-1 and conf_space==""):
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[i,j],[]], [[],[a]]])

    # RAS(h)-1SF-IP
    elif(n_SF==1 and delta_ec==-1 and conf_space=="h"):
        # v(1) unpack to indexing: (ija:aab)
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[i,j],[]], [[],[a]]])
        # v(2) unpack to indexing: (Iia:aab)
        for I in range(ras1):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[I,i],[]], [[],[a]]])
        # v(3) unpack to indexing: (Iijab:aaabb)
        for I in range(ras1):
            for i in range(ras1,ras1+ras2):
                for j in range(ras1,i):
                    for a in range(ras1,ras1+ras2):
                        for b in range(ras1,a):
                            dets_list.append([[[I,i,j],[]], [[],[a,b]]])

    # CAS-EA
    elif(n_SF==0 and delta_ec==1 and conf_space==""):
        for a in range(ras1,ras1+ras2):
            dets_list.append([[[],[]], [[],[a]]])

    # RAS(h)-EA
    elif(n_SF==0 and delta_ec==1 and conf_space=="h"):
        # v(1) indexing: (a:b)
        for a in range(ras1,ras1+ras2):
            dets_list.append([[[],[]], [[],[a]]])
        # v(2) indexing: (Iab:bbb)
        for I in range(ras1):
            for a in range(ras1,ras1+ras2):
                for b in range(ras1,a):
                    dets_list.append([[[],[I]], [[],[a,b]]])

    # doing RAS(p)-EA calculation
    elif(n_SF==0 and delta_ec==1 and conf_space=="p"):
        # v(1) indexing: (a:b)
        for a in range(ras1,ras1+ras2):
            dets_list.append([[[],[]], [[],[a]]])
        # v(2) indexing: (A:b)
        for A in range(ras1+ras2,ras1+ras2+ras3):
            dets_list.append([[[],[]], [[],[A]]])
        # v(3) indexing: (Aia:aab)
        for A in range(ras1+ras2,ras1+ras2+ras3):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[i],[]], [[A],[a]]])

    # CAS-1SF-EA
    elif(n_SF==1 and delta_ec==1 and conf_space==""):
        # v(1) unpack to indexing: (iab:abb)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                for b in range(ras1,a):
                    dets_list.append([[[i],[]], [[],[a,b]]])

    # RAS(p)-1SF-EA
    elif(n_SF==1 and delta_ec==1 and conf_space=="p"):
        # v(1) unpack to indexing: (iab:abb)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                for b in range(ras1,a):
                    dets_list.append([[[i],[]], [[],[a,b]]])
        # v(2) unpack to indexing: (Aia:bab)
        for A in range(ras1+ras2,ras1+ras2+ras3):
            for i in range(ras1,ras1+ras2):
                for a in range(ras1,ras1+ras2):
                    dets_list.append([[[i],[]], [[],[A,a]]])
        # v(3) unpack to indexing: (Aijab:baabb)
        for i in range(ras1,ras1+ras2):
            for j in range(ras1,i):
                for A in range(ras1+ras2,ras1+ras2+ras3):
                    for a in range(ras1,ras1+ras2):
                        for b in range(ras1,a):
                            dets_list.append([[[i,j],[]], [[],[A,a,b]]])

    else:
        print("Sorry, %iSF with electron count change of %i and conf space %s not yet supported." %(n_SF, delta_ec, conf_space) )
        print("Exiting...")
        exit()

    return dets_list

def smp_with_eri(n_SF, delta_ec, conf_space, v, docc, socc, virt):
    # get S**2 ERIs
    s2_tei = TEISpin(docc, socc, virt)

    na = docc + socc
    nb = docc
    nbf = docc + socc + virt
    Fa_tmp_tmp = np.einsum('piqi->pq', s2_tei.full()[:, :na, :, :na]) - np.einsum('piiq->pq', s2_tei.full()[:, :na, :na, :])
    Fb_tmp_tmp = np.einsum('piqi->pq', s2_tei.full()[:, nbf:nbf+nb, :, nbf:nbf+nb])

    #F = (Fa_tmp + Fb_tmp)

    Fa = Fa_tmp_tmp[:nbf, :nbf]
    Fb = Fb_tmp_tmp[nbf:, nbf:]

    #Fa = np.eye(nbf)
    #Fb = 0*np.eye(nbf)

    #print(np.allclose(Fa[:nbf, nbf:], np.zeros(Fa[:nbf, nbf:].shape)))
    #print(np.allclose(Fa[nbf:, :nbf], np.zeros(Fa[nbf:, :nbf].shape)))
    #print(np.allclose(Fa[nbf:, nbf:], np.zeros(Fa[nbf:, nbf:].shape)))
    #print(np.allclose(Fb[:nbf, nbf:], np.zeros(Fb[:nbf, nbf:].shape)))

    #Fa = np.einsum('piqi->pq', s2_tei.full()[:, :, :, :])[:nbf, :nbf] - np.einsum('piiq->pq', s2_tei.full()[:, :, :, :])[:nbf, :nbf]
    #Fb = np.einsum('piqi->pq', s2_tei.full()[:, :, :, :])[nbf:, nbf:]

    # 1SF-CAS
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        v_b1 = np.reshape(v, (socc,socc)) # v for block 1
        #   sig(ia:ba) += -v(ia:ba) (eps(a:b)-eps(i:a))
        Fi_tmp = Fa[docc:docc+socc, docc:docc+socc]
        Fa_tmp = Fb[docc:docc+socc, docc:docc+socc]
        F_tmp = np.einsum("ib,ba->ia", v_b1, Fa_tmp) - np.einsum("ja,ji->ia", v_b1, Fi_tmp)
        F_tmp.shape = (v.shape[0], )
        #   sig(ai:ba) += -v(bj:ba) I(ajbi:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 1, 0, 1, 0)
        tei_tmp = np.reshape(-1.0*np.einsum("jb,ajbi->ia", v_b1, tei_tmp), (v.shape[0], ))
        return np.einsum("i,i->", v, F_tmp + tei_tmp)

    '''
    # 2SF-CAS
    if(n_SF==2 and delta_ec==0 and conf_space==""):
        # v(1) unpack to indexing: (ijab:aabb)
        v_ref1 = np.zeros((socc, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for a in range(socc):
                    for b in range(a):
                        v_ref1[i, j, a, b] = v[index]
                        v_ref1[i, j, b, a] = -1.0*v[index]
                        v_ref1[j, i, a, b] = -1.0*v[index]
                        v_ref1[j, i, b, a] = v[index]
                        index = index + 1
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 0, 1, 0, 1)
        tei_tmp = np.einsum("jkbc,iajb->ikac", v_ref1, tei_tmp)
        return np.einsum("ijab,ijab->", v_ref1, tei_tmp)
    '''

    # RAS(h)-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="h"):
        v_b1 = v[:(socc*socc)] # v for block 1
        v_b2 = v[(socc*socc):((socc*socc)+(socc*docc))] # v for block 2
        v_b3 = v[((socc*socc)+(socc*docc)):] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (socc, socc))
        # v(2) indexing: (Ia:ab)
        v_ref2 = np.reshape(v_b2, (docc, socc))
        # v(3) unpack to indexing: (Iiab:babb)
        v_ref3 = np.zeros((docc, socc, socc, socc))
        index = 0
        for I in range(docc):
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref3[I, i, a, b] = v_b3[index]
                        v_ref3[I, i, b, a] = -1.0*v_b3[index]
                        index = index + 1
        # now do contractions
        #       H(1,1) v(1) = sig(1)
        #   sig(ia:ba) += -v(ia:ba) (eps(a:b)-eps(i:a))
        Fi_tmp = Fa[docc:docc+socc, docc:docc+socc]
        Fa_tmp = Fb[docc:docc+socc, docc:docc+socc]
        sig_1 = np.einsum("ib,ba->ia", v_ref1, Fa_tmp) - np.einsum("ja,ji->ia", v_ref1, Fi_tmp)
        #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 1, 0, 1, 0)
        sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)
        #       H(1,2) v(2) = sig(1)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
        tei_tmp = s2_tei.get_subblock(1, 2, 2, 2, 0, 1, 0, 1)
        sig_1 = sig_1 - np.einsum("Ib,Iaib->ia", v_ref2, tei_tmp)
        #       H(1,3) v(3) = sig(1)
        #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
        tei_tmp = s2_tei.get_subblock(1, 2, 2, 2, 1, 0, 1, 0)
        sig_1 = sig_1 - np.einsum("Ijba,Ijbi->ia", v_ref3, tei_tmp)
        #       H(2,1) v(1) = sig(2)
        #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
        tei_tmp = s2_tei.get_subblock(2, 2, 1, 2, 0, 1, 0, 1)
        sig_2 = -1.0*np.einsum("ib,iaIb->Ia", v_ref1, tei_tmp)
        #       H(2,2) v(2) = sig(2)
        #   sig(Ia:ab) += sig(Ia:ab)*F(ab:bb) - sig(Ja:ab)*F(IJ:aa)
        Fa_tmp = Fa[0:docc, 0:docc]
        Fb_tmp = Fb[docc:docc+socc, docc:docc+socc]
        sig_2 = sig_2 + np.einsum("Ib,ab->Ia", v_ref2, Fb_tmp) - np.einsum("Ja,IJ->Ia", v_ref2, Fa_tmp)
        #   sig(Ia:ab) += v(Jb:ab)*I(JaIb:abab)
        tei_tmp = s2_tei.get_subblock(1, 2, 1, 2, 0, 1, 0, 1)
        sig_2 = sig_2 - np.einsum("Jb,JaIb->Ia", v_ref2, tei_tmp)
        #       H(2,3) v(3) = sig(2)
        #   sig(Ia:ab) += -v(Jiab:babb)*I(JibI:baba)
        tei_tmp = s2_tei.get_subblock(1, 2, 2, 1, 1, 0, 1, 0)
        sig_2 = sig_2 + np.einsum("Jiab,JibI->Ia", v_ref3, tei_tmp)
        #       H(3,1) v(1) = sig(3)
        #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 1, 0, 1, 0, 1)
        sig_3 = np.einsum("ja,jbiI->Iiab", v_ref1, tei_tmp) - np.einsum("jb,jaiI->Iiab", v_ref1, tei_tmp)
        #       H(3,2) v(2) = sig(3)
        #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
        tei_tmp = s2_tei.get_subblock(1, 2, 2, 1, 0, 1, 0, 1)
        sig_3 = sig_3 + np.einsum("Ja,JbiI->Iiab", v_ref2, tei_tmp)
        sig_3 = sig_3 - np.einsum("Jb,JaiI->Iiab", v_ref2, tei_tmp)
        #       H(3,3) v(3) = sig(3)
        #   sig(Iiab:babb) += t(Iiac:babb)*F(bc:bb) - t(Iibc:babb)*F(ac:bb)
        F_tmp = Fb[docc:docc+socc, docc:docc+socc]
        sig_3 = sig_3 + np.einsum("Iiac,bc->Iiab", v_ref3, F_tmp) - np.einsum("Iibc,ac->Iiab", v_ref3, F_tmp)
        #   sig(Iiab:babb) += -1.0*t(Ijab:babb)*F(ij:aa)
        sig_3 = sig_3 - np.einsum("Ijab,ij->Iiab", v_ref3, F_tmp)
        #   sig(Iiab:babb) += -1.0*t(Jiab:babb)*F(IJ:bb)
        F_IJ_tmp = Fa[0:docc, 0:docc]
        sig_3 = sig_3 - np.einsum("Jiab,IJ->Iiab", v_ref3, F_IJ_tmp)
        #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 1, 0, 1, 0)
        sig_3 = sig_3 - np.einsum("Ijcb,ajci->Iiab", v_ref3, tei_tmp) + np.einsum("Ijca,bjci->Iiab", v_ref3, tei_tmp)
        #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
        tei_tmp = s2_tei.get_subblock(1, 2, 1, 2, 1, 0, 1, 0)
        sig_3 = sig_3 + np.einsum("Jjab,JjIi->Iiab", v_ref3, tei_tmp)
        # sigs complete-- free to reshape!
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], ))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], ))
        # pack sig(3) vector for returning
        sig_3_out = np.zeros((v_b3.shape[0], ))
        index = 0
        for I in range(docc):
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        sig_3_out[index] = sig_3[I, i, a, b]
                        index = index + 1
        return np.einsum("i,i->", v_b1, sig_1) + np.einsum("i,i->", v_b2, sig_2) + np.einsum("i,i->", v_b3, sig_3_out) 

    # RAS(p)-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="p"):
        # Separate guess vector into blocks 1, 2, and 3
        v_b1 = v[:(socc*socc)] # v for block 1
        v_b2 = v[(socc*socc):(socc*socc)+(socc*virt)] # v for block 2
        v_b3 = v[(socc*socc)+(socc*virt):] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (socc, socc))
        # v(2) indexing: (Ai:ab)
        v_ref2 = np.reshape(v_b2, (virt, socc))
        # v(3) unpack to indexing: (Aijb:aaab)
        v_ref3 = np.zeros((virt, socc, socc, socc))
        index = 0
        for i in range(socc):
            for j in range(i):
                for A in range(virt):
                    for b in range(socc):
                        v_ref3[A, i, j, b] = v_b3[index]
                        v_ref3[A, j, i, b] = -1.0*v_b3[index]
                        index = index + 1
        #       H(1,1) v(1) = sig(1)
        #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
        Fa_tmp = Fa[docc:docc+socc, docc:docc+socc]
        Fb_tmp = Fb[docc:docc+socc, docc:docc+socc]
        sig_1 = np.einsum("ib,ab->ia", v_ref1, Fb_tmp) - np.einsum("ja,ji->ia", v_ref1, Fa_tmp)
        #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 1, 0, 1, 0)
        sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)
        #       H(1,2) v(2) = sig(1)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 3, 2, 1, 0, 1, 0)
        sig_1 = sig_1 - np.einsum("Bj,ajBi->ia", v_ref2, tei_tmp)
        #       H(1,3) v(3) = sig(1)
        #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 3, 1, 0, 1, 0)
        sig_1 = sig_1 - np.einsum("Aijb,ajbA->ia", v_ref3, tei_tmp)
        #       H(2,1) v(1) = sig(2)
        #   sig(iA:ab) += v(jb:ab)*t(jb:ab)*I(Ajbi:baba)
        tei_tmp = s2_tei.get_subblock(3, 2, 2, 2, 1, 0, 1, 0)
        sig_2 = -1.0*np.einsum("jb,Ajbi->Ai", v_ref1, tei_tmp)
        #       H(2,2) v(2) = sig(2)
        #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
        Fa_tmp = Fa[docc:docc+socc, docc:docc+socc]
        Fb_tmp = Fb[docc+socc:docc+socc+virt, docc+socc:docc+socc+virt]
        sig_2 = np.einsum("Bi,AB->Ai", v_ref2, Fb_tmp) - np.einsum("Aj,ji->Ai", v_ref2, Fa_tmp)
        #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
        tei_tmp = s2_tei.get_subblock(3, 2, 3, 2, 1, 0, 1, 0)
        sig_2 = sig_2 - np.einsum("Bj,AjBi->Ai", v_ref2, tei_tmp)
        #       H(2,3) v(3) = sig(2)
        #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
        tei_tmp = s2_tei.get_subblock(3, 2, 2, 3, 1, 0, 1, 0)
        sig_2 = sig_2 - np.einsum("Bijc,AjcB->Ai", v_ref3, tei_tmp)
        #       H(3,1) v(1) = sig(3)
        #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
        tei_tmp = s2_tei.get_subblock(3, 2, 2, 2, 0, 1, 0, 1)
        sig_3 = -1.0*np.einsum("ic,Abjc->Aijb", v_ref1, tei_tmp) + np.einsum("jc,Abic->Aijb", v_ref1, tei_tmp)
        #       H(3,2) v(2) = sig(3)
        #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
        tei_tmp = s2_tei.get_subblock(3, 2, 2, 3, 0, 1, 0, 1)
        sig_3 = -1.0*np.einsum("Ci,AbjC->Aijb", v_ref2, tei_tmp)
        sig_3 = sig_3 + np.einsum("Cj,AbiC->Aijb", v_ref2, tei_tmp)
        #       H(3,3) v(3) = sig(3)
        #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
        F_bc_tmp = Fb[docc:docc+socc, docc:docc+socc]
        F_AB_tmp = Fb[docc+socc:docc+socc+virt, docc+socc:docc+socc+virt]
        sig_3 = sig_3 + np.einsum("Aijc,bc->Aijb", v_ref3, F_bc_tmp) # no contribution
        sig_3 = sig_3 + np.einsum("Bijb,AB->Aijb", v_ref3, F_AB_tmp) # no contribution
        Fi_tmp = Fa[docc:docc+socc, docc:docc+socc]
        sig_3 = sig_3 - np.einsum("Aikb,kj->Aijb", v_ref3, Fi_tmp) + np.einsum("Ajkb,ki->Aijb", v_ref3, Fi_tmp)
        #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
        tei_tmp = s2_tei.get_subblock(3, 2, 3, 2, 0, 1, 0, 1)
        sig_3 = sig_3 + np.einsum("Bijc,AbBc->Aijb", v_ref3, tei_tmp)
        #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 0, 1, 0, 1)
        sig_3 = sig_3 - np.einsum("Akjc,kbic->Aijb", v_ref3, tei_tmp)
        #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
        sig_3 = sig_3 + np.einsum("Akic,kbjc->Aijb", v_ref3, tei_tmp)
        # sigs complete-- free to reshape!
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], ))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], ))
        # pack sig(3) vector for returning
        sig_3_out = np.zeros((v_b3.shape[0], )) # add 0.5
        index = 0
        for i in range(socc):
            for j in range(i):
                for A in range(virt):
                    for b in range(socc):
                        sig_3_out[index] = sig_3[A, i, j, b]
                        index = index + 1
        return np.einsum("i,i->", v_b1, sig_1) + np.einsum("i,i->", v_b2, sig_2) + np.einsum("i,i->", v_b3, sig_3_out)

    # CAS-IP
    if(n_SF==0 and delta_ec==-1 and (conf_space=="" or conf_space=="p")):
        #F_tmp = F[0:socc, 0:socc]
        #sig_1 = -1.0*np.einsum("j,ji->i", v, F_tmp)
        #return np.einsum("i,i->", v, sig_1)
        return 0

    # CAS-EA
    if(n_SF==0 and delta_ec==1 and (conf_space=="" or conf_space=="h")):
        #F_tmp = F[0:socc, 0:socc]
        #sig_1 = np.einsum("b,ab->a", v, F_tmp)
        #return np.einsum("i,i->", v, sig_1)
        return 0

    # doing RAS(p)-EA calculation
    if(n_SF==0 and delta_ec==1 and conf_space=="p"):
        v_b1 = v[0:socc] # v for block 1
        v_b2 = v[socc:socc+virt] # v for block 2
        v_b3 = v[socc+virt:] # v for block 3
        # v(1) indexing: (a:b)
        v_ref1 = np.reshape(v_b1, (socc))
        # v(2) indexing: (A:b)
        v_ref2 = np.reshape(v_b2, (virt))
        # v(3) indexing: (iAa:aab)
        v_ref3 = np.reshape(v_b3, (virt, socc, socc))
        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        #F_tmp = F[0:virt, 0:virt]
        #sig_2 = np.einsum("B,AB->A", v_ref2, F_tmp)
        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 
        tei_tmp = s2_tei.get_subblock(2, 3, 3, 2, 0, 1, 0, 1)
        sig_2 = np.einsum("Bia,iABa->A", v_ref3, tei_tmp)
        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 
        tei_tmp = s2_tei.get_subblock(3, 2, 2, 3, 0, 1, 0, 1)
        sig_3 = np.einsum("B,AaiB->Aia", v_ref2, tei_tmp)
        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 
        #F_tmp = F[0:socc, 0:socc]
        #sig_3 = sig_3 - np.einsum("Aja,ji->Aia", v_ref3, F_tmp)
        #F_tmp = F[0:virt, 0:virt]
        #sig_3 = sig_3 + np.einsum("Bia,AB->Aia", v_ref3, F_tmp)
        #F_tmp = F[0:socc, 0:socc]
        #sig_3 = sig_3 + np.einsum("Aib,ab->Aia", v_ref3, F_tmp)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 0, 1, 0, 1)
        sig_3 = sig_3 - np.einsum("Ajb,jaib->Aia", v_ref3, tei_tmp)
        tei_tmp = s2_tei.get_subblock(3, 2, 3, 2, 0, 1, 0, 1)
        sig_3 = sig_3 + np.einsum("Bib,AaBb->Aia", v_ref3, tei_tmp)
        # reshape and return
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], ))
        sig_3 = np.reshape(sig_3, (v_b3.shape[0], ))
        return np.einsum("i,i->", v_b2, sig_2) + np.einsum("i,i->", v_b3, sig_3)

    # doing RAS(h)-IP calculation
    if(n_SF==0 and delta_ec==-1 and conf_space=="h"):
        v_b1 = v[0:socc] # v for block 1
        v_b2 = v[socc:docc+socc] # v for block 2
        v_b3 = v[docc+socc:] # v for block 3
        # v(1) indexing: (i:a)
        v_ref1 = np.reshape(v_b1, (socc))
        # v(2) indexing: (I:a)
        v_ref2 = np.reshape(v_b2, (docc))
        # v(3) indexing: (Iia:bba)
        v_ref3 = np.reshape(v_b3, (docc, socc, socc))
        #       H(2,2) v(2) = sig(2)
        #F_tmp = F[0:docc, 0:docc]
        #sig_2 = -1.0*np.einsum("J,JI->I", v_ref2, F_tmp)
        #       H(2,3) v(3) = sig(2)
        tei_tmp = s2_tei.get_subblock(1, 2, 2, 1, 1, 0, 1, 0)
        sig_2 = -1.0*np.einsum("Jia,JiaI->I", v_ref3, tei_tmp)
        #       H(3,2) v(2) = sig(3)
        tei_tmp = s2_tei.get_subblock(2, 1, 1, 2, 1, 0, 1, 0)
        sig_3 = -1.0*np.einsum("J,aJIi->Iia", v_ref2, tei_tmp)
        #       H(3,3) v(3) = sig(3)
        #F_tmp = F[0:socc, 0:socc]
        #sig_3 = sig_3 - np.einsum("Ija,ji->Iia", v_ref3, F_tmp)
        #F_tmp = F[0:socc, 0:socc]
        #sig_3 = sig_3 + np.einsum("Iib,ab->Iia", v_ref3, F_tmp)
        #F_tmp = F[0:docc, 0:docc]
        #sig_3 = sig_3 - np.einsum("Jia,JI->Iia", v_ref3, F_tmp)
        tei_tmp = s2_tei.get_subblock(2, 2, 2, 2, 1, 0, 1, 0)
        sig_3 = sig_3 - np.einsum("Ijb,ajbi->Iia", v_ref3, tei_tmp)
        tei_tmp = s2_tei.get_subblock(1, 2, 1, 2, 1, 0, 1, 0)
        sig_3 = sig_3 + np.einsum("Jja,JjIi->Iia", v_ref3, tei_tmp)

        sig_2 = np.reshape(sig_2, (v_b2.shape[0], ))
        sig_3 = np.reshape(sig_3, (v_b3.shape[0], ))
        return np.einsum("i,i->", v_b2, sig_2) + np.einsum("i,i->", v_b3, sig_3)

    else:
        return 0.0

def print_det_list(n_SF, delta_ec, conf_space, ras1, ras2, ras3):
    print("ROHF Config:")
    print("\tDOCC: %i" %ras1)
    print("\tSOCC: %i" %ras2)
    print("\tVIRT: %i" %ras3)
    det_list = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)
    count = 0
    for i in range(len(det_list)):
        print("DET %i" %i, end='')
        # annihilated
        print("\tREMOVE:\tA: %9s" %(det_list[i][0][0]), end='')
        print("\tB: %9s" %(det_list[i][0][1]), end='')
        print("\tADD:\tB: %9s" %(det_list[i][1][0]), end='')
        print("\tB: %9s" %(det_list[i][1][1]))


# Give a CI vector, prints information about the most important determinants.
# Parameters:
#    n_SF            Number of spin-flips
#    delta_ec        Change in electron count
#    conf_space      Excitation scheme (Options: "", "h", "p", "h,p")
#    n_dets          Number of determinants
#    vect            Eigenvector (CI coefficients)
#    docc            Number of doubly occupied orbitals
#    socc            Number of singly occupied orbitals
#    virt            Number of doubly unoccupied orbitals
#    dets_to_print   Number of determinants to print (optional)
# Returns: nothing
def print_dets(vect, n_SF, delta_ec, conf_space, n_dets, ras1, ras2, ras3, dets_to_print=10):

    # fix dets_to_print if larger than number of dets
    if(dets_to_print > n_dets):
        dets_to_print = n_dets

    # find largest-magnitude contributions
    sort = abs(vect).argsort()[::-1]

    print("Coeff.\t\tImportant MO Occupations")
    # obtain ordered determinant lists
    dets = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)

    for s in sort[:dets_to_print]:
        print("%10.6f " %(vect[s]), end='')
        # annihilated
        print("\tREMOVE:\tA: %9s" %(dets[s][0][0]), end='')
        print("\tB: %9s" %(dets[s][0][1]), end='')
        print("\tADD:\tB: %9s" %(dets[s][1][0]), end='')
        print("\tB: %9s" %(dets[s][1][1]))

