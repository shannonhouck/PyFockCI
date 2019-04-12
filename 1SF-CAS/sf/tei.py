import math
import numpy as np
import psi4

"""
Class for two-electron integral object handling.

Refs:
Psi4NumPy Tutorials
"""

# Class for two-electron integral object handling. Uses Psi4.
class TEI:
    def __init__(self):
        pass

    ''' 
    Returns a given subblock of the ERI matrix.
    '''
    def get_subblock(self, a, b, c, d): 
        pass

# Class for full TEI integrals.
#   np_tei         Allows for importing a previously-computed TEI integral (in NumPy martrix form)
class TEIFull(TEI):
    def __init__(self, C, basis, ras1, ras2, ras3, np_tei=None):
        if(not type(np_tei)==type(None)):
            print("Reading in two-electron integrals...")
            self.eri = np_tei
        else:
            # get necessary integrals/matrices from Psi4 (AO basis)
            mints = psi4.core.MintsHelper(basis)
            self.eri = psi4.core.Matrix.to_array(mints.ao_eri())
            # put in physicists' notation
            self.eri = self.eri.transpose(0, 2, 1, 3)
            # move to MO basis
            self.eri = np.einsum('pqrs,pa',self.eri,C)
            self.eri = np.einsum('aqrs,qb',self.eri,C)
            self.eri = np.einsum('abrs,rc',self.eri,C)
            self.eri = np.einsum('abcs,sd',self.eri,C)
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[ras1+ras2,ras1+ras2+ras3]]

    def get_subblock(self, a, b, c, d):
        return self.eri[self.ind[a][0]:self.ind[a][1], self.ind[b][0]:self.ind[b][1], self.ind[c][0]:self.ind[c][1], self.ind[d][0]:self.ind[d][1]]

    def get_full(self):
        return self.eri

# Class for full TEI integrals.
class TEIDF(TEI):
    # Used Psi4NumPy for reference for this section
    # ras1, ras2, ras3 indicate the number of orbitals in each section
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space, np_tei=None, np_J=None):
        if(not type(np_tei)==type(None)):
            eri = np_tei
            J = np_J
        else:
            # get info from Psi4
            zero = psi4.core.BasisSet.zero_ao_basis_set()
            mints = psi4.core.MintsHelper(basis)
            # (Q|pq)
            eri = psi4.core.Matrix.to_array(mints.ao_eri(zero, aux, basis, basis))
            eri = np.squeeze(eri)
            C = psi4.core.Matrix.to_array(C)
            # set up J^-1/2 (don't need to keep)
            J = mints.ao_eri(zero, aux, zero, aux)
            J.power(-0.5, 1e-14)
            J = np.squeeze(J)
        # Contract and obtain final form
        eri = np.einsum("PQ,Qpq->Ppq", J, eri)
        # set up C
        C_ras1 = C[:, 0:ras1]
        C_ras2 = C[:, ras1:ras1+ras2]
        C_ras3 = C[:, ras1+ras2:]
        # move to MO basis
        # Notation: ij in active space, IJ in docc, AB in virtual
        # Bnm notation, where n and m indicate RAS space (1/2/3)
        # all of them need RAS2
        B2m = np.einsum('Ppq,pi->Piq', eri, C_ras2)
        self.B22 = np.einsum('Piq,qj->Pij', B2m, C_ras2)
        # if configuration space is "h"
        if(conf_space == "h"):
            B1m = np.einsum('Ppq,pI->PIq', eri, C_ras1)
            self.B11 = np.einsum('PIq,qJ->PIJ', B1m, C_ras1)
            self.B12 = np.einsum('PIq,qj->PIj', B1m, C_ras2)
            self.B21 = np.einsum('Piq,qJ->PiJ', B2m, C_ras1)
        if(conf_space == "p"):
            B3m = np.einsum('Ppq,pA->PAq', eri, C_ras3)
            self.B33 = np.einsum('PAq,qB->PAB', B3m, C_ras3)
            self.B32 = np.einsum('PAq,qj->PAj', B3m, C_ras2)
            self.B23 = np.einsum('Piq,qA->PiA', B2m, C_ras3)
        if(conf_space == "h,p"):
            B1m = np.einsum('Ppq,pI->PIq', eri, C_ras1)
            self.B11 = np.einsum('PIq,qJ->PIJ', B1m, C_ras1)
            self.B12 = np.einsum('PIq,qj->PIj', B1m, C_ras2)
            self.B21 = np.einsum('Piq,qJ->PiJ', B2m, C_ras1)
            self.B13 = np.einsum('PIq,qA->PIA', B1m, C_ras3)
            B3m = np.einsum('Ppq,pA->PAq', eri, C_ras3)
            self.B33 = np.einsum('PAq,qB->PAB', B3m, C_ras3)
            self.B32 = np.einsum('PAq,qj->PAj', B3m, C_ras2)
            self.B31 = np.einsum('PAq,qJ->PAJ', B3m, C_ras1)
            self.B23 = np.einsum('Piq,qA->PiA', B2m, C_ras3)

    # a, b, c, d defined as RAS(1/2/3) spaces, integers
    def get_subblock(self, a, b, c, d):
        # the B matrices are still in chemists' notation, don't switch b/c until end
        if(a == 1):
            if(c == 1):
                B_bra = self.B11
            if(c == 2):
                B_bra = self.B12
            if(c == 3):
                B_bra = self.B13
        if(a == 2):
            if(c == 1):
                B_bra = self.B21
            if(c == 2):
                B_bra = self.B22
            if(c == 3):
                B_bra = self.B23
        if(a == 3):
            if(c == 1):
                B_bra = self.B31
            if(c == 2):
                B_bra = self.B32
            if(c == 3):
                B_bra = self.B33

        if(b == 1):
            if(d == 1):
                B_ket = self.B11
            if(d == 2):
                B_ket = self.B12
            if(d == 3):
                B_ket = self.B13
        if(b == 2):
            if(d == 1):
                B_ket = self.B21
            if(d == 2):
                B_ket = self.B22
            if(d == 3):
                B_ket = self.B23
        if(b == 3):
            if(d == 1):
                B_ket = self.B31
            if(d == 2):
                B_ket = self.B32
            if(d == 3): 
                B_ket = self.B33

        return np.einsum("Pij,Pkl->ijkl", B_bra, B_ket).transpose(0, 2, 1, 3)

# Class for full TEI integrals.
class TEISpin(TEI):
    def __init__(self, ras1, ras2, ras3):
        self.nbf = ras1 + ras2 + ras3
        nbf = ras1 + ras2 + ras3
        self.eri = np.zeros((2*nbf, 2*nbf, 2*nbf, 2*nbf))
        for i in range(nbf):
            for j in range(nbf):
                #self.eri[i+nbf, j, i, j+nbf] = 0
                self.eri[i+nbf, j, i, j+nbf] = -1.0
                self.eri[i+nbf, j, j+nbf, i] = -1.0
                self.eri[i, j+nbf, j+nbf, i] = -1.0
                self.eri[i, j+nbf, i, j+nbf] = -1.0
                #self.eri[i, j+nbf, i+nbf, j] = -1.0
                #self.eri[i+nbf, j, j+nbf, i] = -1.0
                #self.eri[i, j+nbf, j, i+nbf] = -1.0
                #self.eri[i+nbf, j, i+nbf, j] = 1.0
                #self.eri[i, j+nbf, i, j+nbf] = 1.0
        self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[ras1+ras2,ras1+ras2+ras3]]

    # s1, s2, s3, s4    Alpha or beta spin (0=alpha, 1=beta)
    def get_subblock(self, a, b, c, d, s1, s2, s3, s4): 
        nbf = self.nbf
        return self.eri[(s1*nbf)+self.ind[a][0]:(s1*nbf)+self.ind[a][1], (s2*nbf)+self.ind[b][0]:(s2*nbf)+self.ind[b][1],
                        (s3*nbf)+self.ind[c][0]:(s3*nbf)+self.ind[c][1], (s4*nbf)+self.ind[d][0]:(s4*nbf)+self.ind[d][1]]

    def full(self):
        return self.eri

    def get_alpha(self):
        nbf = self.nbf
        return self.eri[:nbf, :nbf, :nbf, :nbf]

    def get_beta(self):
        nbf = self.nbf
        return self.eri[nbf:, nbf:, nbf:, nbf:]

