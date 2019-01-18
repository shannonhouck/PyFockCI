import numpy as np

# Calculates S**2
# Parameters:
#    n_SF            Number of spin-flips
#    delta_ec        Change in electron count
#    conf_space      Excitation scheme (Options: "", "h", "p", "h,p")
#    vect            Eigenvector (CI coefficients)
#    docc            Number of doubly occupied orbitals
#    socc            Number of singly occupied orbitals
#    virt            Number of doubly unoccupied orbitals
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
        s2 = s2 + np.einsum("Ia,Ia->", v_ref2, v_ref2)
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    s2 = s2 - 2.0*v_ref2[I,a]*v_ref3[I,p,a,p]
        # block 3
        for I in range(docc):
            for a in range(socc):
                for p in range(socc):
                    for q in range(socc):
                        s2 = s2 + v_ref3[I,p,a,p]*v_ref3[I,q,a,q]
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
        return 0.0


