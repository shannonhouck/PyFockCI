import math
import numpy as np
import time
import psi4

"""
Class for two-electron integral object handling.

This class handles the two-electron integrals. It generates and stores 
integrals in the form of NumPy arrays. Relevant sub-blocks can be easily 
accessed via the ``get_subblock`` routine.

Refs:
* Psi4NumPy Tutorials
"""

class TEI:
    """General parent class for two-electron integral object handling.

    This class handles the two-electron integrals. It generates and stores 
    integrals in the form of NumPy arrays. Relevant sub-blocks can be easily 
    accessed via the ``get_subblock`` routine.
    """
    def __init__(self):
        """Initialize TEI object.
        """
        pass

    def get_subblock(self, a, b, c, d): 
        """Returns a given subblock of the ERI matrix.
        """
        pass

class TEIFullBase(TEI):
    """Base class for constructing full TEI integrals.
    """
    def __init__(self, ras1, ras2, ras3, conf_space, np_tei):
        """Initialize TEI object.
        """
        pass

    def get_subblock(self, a, b, c, d):
        """Returns a given subblock of the two-electron integrals.
           The RAS space to return is given by a, b, c, and d, and 
           the returned integral has the form <ab|cd>.
           So to get the block with a and c in RAS1 and b and d in RAS2,
           one would use get_subblock(1, 2, 1, 2).
           :param a: RAS space of index 1
           :param b: RAS space of index 2
           :param c: RAS space of index 3
           :param d: RAS space of index 4
           :return: Desired subblock (NumPy array)
        """
        return self.eri[self.ind[a][0]:self.ind[a][1],
                        self.ind[b][0]:self.ind[b][1],
                        self.ind[c][0]:self.ind[c][1],
                        self.ind[d][0]:self.ind[d][1]]

    def get_full(self):
        """Returns the full set of two-electron integrals as a NumPy array.
        """
        return self.eri

class TEIFullNumPy(TEIFullBase):
    """Class for constructing full TEI integrals from a NumPy array.

    This class stores the integrals as a NumPy array, given a NumPy array.
    Note that the relevant subset of the array should be given. So, in the 
    case of a RAS(h) calculation, one would give the TEI constructed in the 
    basis of RAS1 and RAS2 orbitals only (not RAS3).
    """
    def __init__(self, ras1, ras2, ras3, conf_space, np_tei):
        """Initialize TEI object.

        Parameters
        ----------
        ras1 (int): RAS1 orbitals
        ras2 (int): RAS2 orbitals
        ras3 (int): RAS3 orbitals
        np_tei: NumPy array for previously-constructed integrals.
                      This allows us to avoid integral construction.
        Returns
        -------
        Initialized TEI object
        """
        tei_start_time = time.time()
        print("Reading in two-electron integrals...")
        self.eri = np_tei
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        if(conf_space == ""):
            self.ind = [[0,0],[0,0],[0,ras2],[0,0]] # only worry about RAS2 block
        if(conf_space == "h"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[0,0]] # only worry about RAS1+2
        if(conf_space == "p"):
            self.ind = [[0,0],[0,0],[0,ras2],[ras2, ras2+ras3]] # only worry about RAS2+3
        if(conf_space == "h,p"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],
                        [ras1+ras2,ras1+ras2+ras3]]
        print("Constructed TEI object in %i seconds." %(time.time() - tei_start_time))

class TEIFullPsi4(TEIFullBase):
    """Class for constructing full TEI integrals using Psi4.

    This class constructs the full two-electron integrals using Psi4. 
    It then stores the integrals as NumPy arrays.

    """
    def __init__(self, C, basis, ras1, ras2, ras3, conf_space):
        """Initialize two-electron integral object.

        This initializes a TEI object for the given basis set using Psi4. 
        The TEI is stored as a NumPy array and the 

        Parameters
        ----------
        C (numpy array): MO coefficient matrix
        basis (psi4.core.BasisSet): Basis set
        ras1 (int): RAS1 orbitals
        ras2 (int): RAS2 orbitals
        ras3 (int): RAS3 orbitals
        conf_space (string): Hole/particle excitations

        Returns
        -------
        Initialized TEI object
        """
        tei_start_time = time.time()
        # truncate C as needed
        C_np = psi4.core.Matrix.to_array(C, copy=True)
        ras1_C = C_np[:, :ras1]
        ras2_C = C_np[:, ras1:ras1+ras2]
        ras3_C = C_np[:, ras1+ras2:]
        if(conf_space == ""):
            C_act = ras2_C
        elif(conf_space == "h"):
            C_act = np.column_stack((ras1_C, ras2_C))
        elif(conf_space == "p"):
            C_act = np.column_stack((ras2_C, ras3_C))
        elif(conf_space == "h,p"):
            C_act = np.column_stack((ras1_C, ras2_C, ras3_C))
        # get necessary integrals/matrices from Psi4 (AO basis)
        mints = psi4.core.MintsHelper(basis)
        C_act = psi4.core.Matrix.from_array(C_act)
        self.eri = psi4.core.Matrix.to_array(mints.mo_eri(C_act,C_act,C_act,C_act))
        # put in physicists' notation
        self.eri = self.eri.transpose(0, 2, 1, 3)
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        if(conf_space == ""):
            self.ind = [[0,0],[0,0],[0,ras2],[0,0]] # only worry about RAS2 block
        if(conf_space == "h"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[0,0]] # only worry about RAS1+2
        if(conf_space == "p"):
            self.ind = [[0,0],[0,0],[0,ras2],[ras2, ras2+ras3]] # only worry about RAS2+3
        if(conf_space == "h,p"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],
                        [ras1+ras2,ras1+ras2+ras3]]
        print("Constructed TEI object in %i seconds." %(time.time() - tei_start_time))

# DF TEI INTEGRALS

class TEIDFBase(TEI):
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space,
                 np_tei, np_J):
        pass

    def get_subblock(self, a, b, c, d):
        """Returns a given subblock of the two-electron integrals (DF).

           Returns a given subblock of the DF two-electron integral object.
           The RAS space to return is given by a, b, c, and d, and 
           the returned integral has the form <ab|cd>.

           :param a: RAS space of index 1
           :param b: RAS space of index 2
           :param c: RAS space of index 3
           :param d: RAS space of index 4
           :return: Desired subblock (NumPy array)
        """
        # the B matrices are still in chemists' notation
        # don't switch b and c index until end
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

class TEIDFNumPy(TEIDFBase):
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space,
                 np_tei, np_J):
        eri = np_tei
        J = np_J
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
        print("Constructed TEI object in %i seconds." %(time.time() - tei_start_time))

class TEIDFPsi4(TEIDFBase):
    # Used Psi4NumPy for reference for this section
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space,
                 np_tei=None, np_J=None):
        """Initialize density-fitted TEI object.

           :param C: MO coefficient matrix (NumPy array)
           :param basis: Basis set object (Psi4 BasisSet)
           :param ras1: RAS1 orbitals (int)
           :param ras2: RAS2 orbitals (int)
           :param ras3: RAS3 orbitals (int)
           :param ref_method: Program to use to generate TEIs
           :param np_tei: NumPy array for previously-constructed integrals.
                     This allows us to avoid integral construction.
           :param J_tei: Previously-constructed J matrix (NumPy)
           :return: Initialized TEI object
        """
        print("Inititalizing DF-TEI Object....")
        tei_start_time = time.time()
        # get info from Psi4
        zero = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(basis)
        # (Q|pq)
        eri = psi4.core.Matrix.to_array(
              mints.ao_eri(zero, aux, basis, basis))
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
        print("Constructed TEI object in %i seconds." %(time.time() - tei_start_time))


