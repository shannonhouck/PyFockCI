from __future__ import print_function
import numpy as np
from .tei import *

"""
These are functions to handle post-CI analysis, primarily the S**2 values
and information about the determinants.
"""

def calc_sz(n_SF, delta_ec, conf_space, vect, docc, socc, virt):
    # obtain Sz and Sz (same general formula regardless of method)
    na = docc + socc - n_SF
    nb = docc + n_SF
    if(delta_ec==1): # IP
        nb = nb + 1
    if(delta_ec==-1): # EA
        na = na - 1
    det_list = generate_dets(n_SF, delta_ec, conf_space, docc, socc, virt)
    # construct Sz*Sz
    sz_final = 0
    for i, v in enumerate(vect):
        # grab determinants
        det = det_list[i]
        rem_a = det[0][0]
        rem_b = det[0][1]
        add_a = det[1][0]
        add_b = det[1][1]
        # construct Sz for each orbital
        sz_vect = np.zeros(docc+socc+virt)
        # RAS 1 orbitals
        for o in range(docc):
            if o in rem_a:
                if o not in rem_b:
                    sz_vect[o] = -0.5
            if o in rem_b:
                if o not in rem_a:
                    sz_vect[o] = 0.5
        # RAS 2
        for o in range(docc, docc+socc):
            sz_vect[o] = 0.5
            if o in rem_a:
                if o not in add_b:
                    sz_vect[o] = 0.0
            if o in add_a:
                if o not in add_b:
                    sz_vect[o] = 0.5
            if o in add_b:
                if o not in rem_a:
                    sz_vect[o] = 0.0
                if o in rem_a:
                    sz_vect[o] = -0.5
        # RAS 3
        for o in range(docc+socc, docc+socc+virt):
            if o in add_a:
                if o not in add_b:
                    sz_vect[o] = 0.5
            if o in add_b:
                if o not in add_a:
                    sz_vect[o] = -0.5
        sz_final = sz_final + v*v*np.sum(sz_vect)
    return sz_final

def calc_s2(n_SF, delta_ec, conf_space, vect, docc, socc, virt):
    """Calculates S**2 for a given CI vector.
       Input
           n_SF -- Number of spin-flips
           delta_ec -- Change in electron count
           conf_space -- Excitation scheme (Options: "", "h", "p", "h,p")
           vect -- Eigenvector (CI coefficients)
           docc -- Number of doubly occupied orbitals
           socc -- Number of singly occupied orbitals
           virt -- Number of doubly unoccupied orbitals
       Returns
           s2 -- The S**2 expectation value for the state
    """
    s2 = 0.0
    # obtain Sz and Sz (same general formula regardless of method)
    na = docc + socc - n_SF
    nb = docc + n_SF
    if(delta_ec==1): # IP
        nb = nb + 1
    if(delta_ec==-1): # EA
        na = na - 1
    det_list = generate_dets(n_SF, delta_ec, conf_space, docc, socc, virt)
    # construct Sz*Sz
    for i, v in enumerate(vect):
        # grab determinants
        det = det_list[i]
        rem_a = det[0][0]
        rem_b = det[0][1]
        add_a = det[1][0]
        add_b = det[1][1]
        # construct Sz for each orbital
        sz_vect = np.zeros(docc+socc+virt)
        # RAS 1 orbitals
        for o in range(docc):
            if o in rem_a:
                if o not in rem_b:
                    sz_vect[o] = -0.5
            if o in rem_b:
                if o not in rem_a:
                    sz_vect[o] = 0.5
        # RAS 2
        for o in range(docc, docc+socc):
            sz_vect[o] = 0.5
            if o in rem_a:
                if o not in add_b:
                    sz_vect[o] = 0.0
            if o in add_a:
                if o not in add_b:
                    sz_vect[o] = 0.5
            if o in add_b:
                if o not in rem_a:
                    sz_vect[o] = 0.0
                if o in rem_a:
                    sz_vect[o] = -0.5
        # RAS 3
        for o in range(docc+socc, docc+socc+virt):
            if o in add_a:
                if o not in add_b:
                    sz_vect[o] = 0.5
            if o in add_b:
                if o not in add_a:
                    sz_vect[o] = -0.5
        # now evaluate Sz*Sz (over all orbitals i and j)
        tmp = 0.0
        for i in range(docc+socc+virt):
            for j in range(docc+socc+virt):
                if(i != j):
                    tmp = tmp + sz_vect[i]*sz_vect[j]
                else:
                    if(abs(sz_vect[i]) == 0.5):
                        tmp = tmp + 0.75
        s2 = s2 + v*v*tmp

    '''
    OLD CODE (maybe useful for future??)
    for i, v in enumerate(vect):
        # do Sz
        #s2 = s2 + v*v*(0.5*(na) - 0.5*(nb))

        # do Sz^2
        #s2 = s2 + 2.0*v*v*(0.25*(na*na) + 0.25*(nb*nb) - 0.5*(na*nb))
    '''

    # CAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        vect = np.reshape(vect, (socc,socc))
        # S+S-
        for p in range(socc):
            for q in range(socc):
                if(p != q):
                    s2 = s2 + 0.5*vect[p,p]*vect[q,q]
        # S-S+
        for p in range(socc):
            for q in range(socc):
                if(p != q):
                    s2 = s2 + 0.5*vect[p,p]*vect[q,q]
        return s2

    # CAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space=="neutral"):
        # S+S-
        for p in range(socc):
            for q in range(socc):
                if(p != q):
                    s2 = s2 + 0.5*vect[p]*vect[q]
        # S-S+
        for p in range(socc):
            for q in range(socc):
                if(p != q):
                    s2 = s2 + 0.5*vect[p]*vect[q]
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
                        if(p != q):
                            s2 = s2 + v_ref1[q,j,q,b]*v_ref1[p,j,p,b]
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
        for I in range(docc):
            for p in range(socc):
                s2 = s2 + v_ref2[I]*v_ref3[I,p,p]
        # block 3
        for I in range(docc):
            for p in range(socc):
                s2 = s2 + v_ref2[I]*v_ref3[I,p,p]
                for q in range(socc):
                    if(p != q):
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
        for A in range(virt):
            for p in range(socc):
                s2 = s2 - v_ref2[A]*v_ref3[A,p,p]
        # block 3
        for A in range(virt):
            for p in range(socc):
                s2 = s2 - v_ref2[A]*v_ref3[A,p,p]
                for q in range(socc):
                    if(p != q):
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
                    if(p != q):
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
                    if(p != q):
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
                    if(p != q):
                        s2 = s2 + v_ref1[q,i,q]*v_ref1[p,i,p]
        # block 2
        for I in range(docc):
            for p in range(socc):
                for q in range(socc):
                    if(p != q):
                        s2 = s2 + v_ref2[I,p,p]*v_ref2[I,q,q]
        # block 2 w/ block 3
        for I in range(docc):
            for p in range(socc):
                for a in range(socc):
                    for i in range(socc):
                        s2 = s2 - v_ref2[I,i,a]*v_ref3[I,p,i,a,p]
        # block 3
        for I in range(docc):
            for p in range(socc):
                for a in range(socc):
                    for i in range(socc):
                        s2 = s2 - v_ref2[I,i,a]*v_ref3[I,p,i,a,p]
                        for q in range(socc):
                            if(p != q):
                                s2 = s2 + v_ref3[I,q,i,a,q]*v_ref3[I,p,i,a,p]
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
                    if(p != q):
                        s2 = s2 + v_ref1[q,q,a]*v_ref1[p,p,a]
        # block 2
        for A in range(virt):
            for p in range(socc):
                for q in range(socc):
                    if(p != q):
                        s2 = s2 + v_ref2[A,p,p]*v_ref2[A,q,q]
        # block 2
        for A in range(virt):
            for p in range(socc):
                for a in range(socc):
                    for i in range(socc):
                        s2 = s2 + v_ref2[A,i,a]*v_ref3[A,i,p,p,a]
        # block 3
        for A in range(virt):
            for p in range(socc):
                for a in range(socc):
                    for i in range(socc):
                        s2 = s2 + v_ref2[A,i,a]*v_ref3[A,i,p,p,a]
                        for q in range(socc):
                            if(p != q):
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
                if(p != q):
                    s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - v_ref2[I,a]*v_ref3[I,p,a,p]
        # block 3
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - v_ref2[I,a]*v_ref3[I,p,a,p]
                    for q in range(socc):
                        if(p != q):
                            s2 = s2 + v_ref3[I,p,a,p]*v_ref3[I,q,a,q]
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
        v_b3 = vect[n_b1_dets+n_b2_dets:n_b1_dets+n_b2_dets+n_b3_dets] # b3
        v_b4 = vect[n_b1_dets+n_b2_dets+n_b3_dets:
                    n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets] # v for block 4
        v_b5 = vect[n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets:
                    n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets+n_b5_dets] # b5
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
        # block 1 (CAS-SF)
        for p in range(socc):
            for q in range(socc):
                if(p != q):
                    s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2 (h)
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - v_ref2[I,a]*v_ref4[I,p,a,p]
        # block 3 (p)
        for A in range(virt):
            for i in range(socc):
                for p in range(socc):
                    s2 = s2 + v_ref3[A,i]*v_ref5[A,i,p,p]
        # block 4 (h)
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - v_ref2[I,a]*v_ref4[I,p,a,p]
                    for q in range(socc):
                        if(p != q):
                            s2 = s2 + v_ref4[I,p,a,p]*v_ref4[I,q,a,q]
        # block 5 (p)
        for A in range(virt):
            for i in range(socc):
                for q in range(socc):
                    s2 = s2 + v_ref3[A,i]*v_ref5[A,i,q,q]
                    for p in range(socc):
                        if(p != q):
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
                if(p != q):
                    s2 = s2 + v_ref1[p,p]*v_ref1[q,q]
        # block 2
        for A in range(virt):
            for i in range(socc):
                for p in range(socc):
                    s2 = s2 + v_ref2[A,i]*v_ref3[A,i,p,p]
        # block 3
        for A in range(virt):
            for i in range(socc):
                for q in range(socc):
                    s2 = s2 + v_ref2[A,i]*v_ref3[A,i,q,q]
                    for p in range(socc):
                        if(p != q):
                            s2 = s2 + v_ref3[A,i,q,q]*v_ref3[A,i,p,p]
        return s2

    else:
        print("Warning: S**2 not yet supported for this scheme!")
        return s2

def generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3):
    """Returns ordered list of all determinants in the following form:
       det = [...] (det[0] is 0th determinant, det[1] is 1st, etc.)
       det[count] = [[[elim (alpha)], [elim (beta)]], [[add (a)], [add (b)]]]
       Indexing starts at zero.
       Input
           n_SF -- Number of spin-flips
           delta_ec -- Change in electron count
           conf_space -- Configuration space
           ras1 -- Number of RAS1 orbitals
           ras2 -- Number of RAS2 orbitals
           ras3 -- Number of RAS3 orbitals
       Returns
           dets_list -- List of determinants
   """

    # storing list of determinants
    dets_list = []

    # CAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        # make array with determinant data (i->a)
        for i in range(ras1,ras1+ras2):
            for a in range(ras1,ras1+ras2):
                dets_list.append([[[i],[]], [[],[a]]])
        return dets_list

    # CAS-1SF (neutral)
    if(n_SF==1 and delta_ec==0 and conf_space=="neutral"):
        # make array with determinant data (i->i)
        for i in range(ras1,ras1+ras2):
            dets_list.append([[[i],[]], [[],[i]]])
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
                            dets_list.append([[[i,j],[I]], [[],[a,b]]])

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
                            dets_list.append([[[i,j],[]], [[A],[a,b]]])

    else:
        print("Sorry, %iSF with electron count change of %i and conf space \
               %s not yet supported." %(n_SF, delta_ec, conf_space) )
        print("Exiting...")
        exit()

    return dets_list

def print_det_list(n_SF, delta_ec, conf_space, ras1, ras2, ras3):
    """Prints the full list of determinants to standard out.
       This is useful for post-CI analysis.
       Input
           n_SF -- Number of spin-flips
           delta_ec -- Change in electron count
           conf_space -- Configuration space
           ras1 -- Number of RAS1 orbitals
           ras2 -- Number of RAS2 orbitals
           ras3 -- Number of RAS3 orbitals
    """
    print("ROHF Config:")
    print("\tDOCC: %i" %ras1)
    print("\tSOCC: %i" %ras2)
    print("\tVIRT: %i" %ras3)
    # generate list of determinants (start ordering at 1)
    det_list = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)
    count = 0
    for i in range(len(det_list)):
        print("DET %i" %i, end='')
        # annihilated
        print("\tREMOVE:\tA: %9s" %(det_list[i][0][0]), end='')
        print("\tB: %9s" %(det_list[i][0][1]), end='')
        print("\tADD:\tA: %9s" %(det_list[i][1][0]), end='')
        print("\tB: %9s" %(det_list[i][1][1]))


def print_dets(vect, n_SF, delta_ec, conf_space, ras1, ras2, ras3,
               dets_to_print=10):
    """Given a CI vector, prints information about the most important
       determinants.
       Input
           vect -- CI vector
           n_SF -- Number of spin-flips
           delta_ec -- Change in electron count
           conf_space -- Configuration space
           ras1 -- Number of RAS1 orbitals
           ras2 -- Number of RAS2 orbitals
           ras3 -- Number of RAS3 orbitals
           dets_to_print -- Number of determinants to print
    """

    # find largest-magnitude contributions
    sort = abs(vect).argsort()[::-1]
    # obtain ordered determinant lists, start order at 1
    dets = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)
    # fix dets_to_print if larger than number of dets
    if(dets_to_print > len(dets)):
        dets_to_print = len(dets)
        
    # print appropriately
    print("Coeff.\t\tImportant MO Occupations")
    for s in sort[:dets_to_print]:
        print("%10.6f " %(vect[s]), end='')
        # annihilated
        print("\tREMOVE:\tA: %9s" %(dets[s][0][0]), end='')
        print("\tB: %9s" %(dets[s][0][1]), end='')
        print("\tADD:\tA: %9s" %(dets[s][1][0]), end='')
        print("\tB: %9s" %(dets[s][1][1]))

