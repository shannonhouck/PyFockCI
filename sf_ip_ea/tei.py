"""
Two-electron integral object handling.

This module handles the two-electron integrals. It generates and stores 
integrals in the form of NumPy arrays. The TEI subclasses handle 
full (TEIFullBase) and density-fitted integrals (TEIDFBase), 
and both of these have subclasses which handle the multiple ways of 
constructing the integrals (by using Psi4, by passing in pre-constructed 
NumPy arrays, and so on). When an interface to a new program is added, 
a corresponding TEI object specific to that program should be added here 
as a subclass of TEIFullBase or TEIDFBase.

Used Psi4NumPy Tutorials for reference for the density fitting.
"""

import math
import numpy as np
import time
import psi4
from abc import ABCMeta, abstractmethod

class TEI:
    """
    Abstract parent class for two-electron integral object handling.

    This class handles the two-electron integrals. It generates and stores 
    integrals in the form of NumPy arrays. Relevant sub-blocks can be easily 
    accessed via the ``get_subblock`` routine.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Initialize TEI object.
        """
        pass

    @abstractmethod
    def get_subblock(self, a, b, c, d): 
        """Returns a given subblock of the ERI matrix.
        """
        pass

# FULL TEI INTEGRALS

class TEIFullBase(TEI):
    """Base class for constructing full two-electron integrals.
    """
    @abstractmethod
    def __init__(self):
        """Initialize full TEI object (abstract).
        """
        pass

    def get_subblock(self, a, b, c, d):
        """
        Returns a given subblock of the two-electron integrals.

        This returns a NumPy representation of a given subblock of the 
        full two-electron integrals. The subblock to return is given by 
        parameters ``a``, ``b``, ``c``, and ``d``, where each indicates 
        a RAS space (1, 2, or 3). The returned integral is in physicists' 
        notation and has the form <ab|cd>.

        So to get the block with a and c in RAS1 and b and d in RAS2,
        one would use get_subblock(1, 2, 1, 2).

        Parameters
        ----------
        a : int
            RAS space of index 1.
        b : int 
            RAS space of index 2.
        c : int
            RAS space of index 3.
        d : int
            RAS space of index 4.

        Returns
        -------
        numpy.ndarray
            NumPy representation of the desired subblock.
        """
        return self.eri[self.ind[a][0]:self.ind[a][1],
                        self.ind[b][0]:self.ind[b][1],
                        self.ind[c][0]:self.ind[c][1],
                        self.ind[d][0]:self.ind[d][1]]

    def get_full(self):
        """
        Returns the full set of two-electron integrals as a NumPy array.

        This returns the full two-electron integral (for the necessary 
        RAS spaces, given the excitations) as a NumPy array. This is 
        useful for storing the array for use in future calculations.

        Returns
        -------
        numpy.ndarray
            NumPy representation of two-electron integral object.
        """
        return self.eri

class TEIFullNumPy(TEIFullBase):
    """
    Class for constructing full TEI integrals from a NumPy array.

    This class stores the integrals as a NumPy array, given a NumPy array.
    Note that the relevant subset of the array should be given. So, in the 
    case of a RAS(h) calculation, one would give the TEI constructed in the 
    basis of RAS1 and RAS2 orbitals only (not RAS3).

    Attributes
    ----------
    eri : numpy.ndarray
        Numpy representation of the relevant two-electron integrals.
    ind : list
        A list of index ranges for the subblocks (RAS1, RAS2, RAS3).

    """
    def __init__(self, ras1, ras2, ras3, conf_space, np_tei):
        """
        Initialize TEIFullNumPy object.

        This initializes the full NumPy two-electron integral object.
        The TEI is stored as a NumPy array in physicists' notation.

        Parameters
        ----------
        ras1 : int 
            Number of RAS1 orbitals
        ras2 : int 
            Number of RAS2 orbitals
        ras3 : int 
            Number of RAS3 orbitals
        np_tei : numpy.ndarray
            A NumPy array containing previously-constructed integrals.
            This allows us to avoid integral construction. This should be 
            truncated to contain only the relevant subspaces.

        Returns
        -------
        TEIFullNumPy
            Initialized TEI object
        """
        tei_start_time = time.time()
        print("Reading in two-electron integrals...")
        self.eri = np_tei
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        if(conf_space == ""):
            self.ind = [[0,0],[0,0],[0,ras2],[0,0]] 
        if(conf_space == "h"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[0,0]]
        if(conf_space == "p"):
            self.ind = [[0,0],[0,0],[0,ras2],[ras2, ras2+ras3]] 
        if(conf_space == "h,p"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],
                        [ras1+ras2,ras1+ras2+ras3]]
        print("Constructed TEI object in %i seconds." 
              %(time.time() - tei_start_time))

class TEIFullPsi4(TEIFullBase):
    """
    Class for constructing full TEI integrals using Psi4.

    This class constructs the full two-electron integrals using Psi4. 
    It then stores the integrals as NumPy arrays.

    Attributes
    ----------
    eri : numpy.ndarray
        Numpy representation of the relevant two-electron integrals.
    ind : list
        A list of index ranges for the subblocks (RAS1, RAS2, RAS3).
    """
    def __init__(self, C, basis, ras1, ras2, ras3, conf_space):
        """
        Initialize two-electron integral object.

        This initializes a TEI object for the given basis set using Psi4. 
        The TEI is stored as a NumPy array in physicists' notation.

        Parameters
        ----------
        C : numpy.ndarray
            MO coefficient matrix.
        basis : psi4.core.BasisSet
            Psi4 basis set object to use for integral construction.
        ras1 : int 
            Number of RAS1 orbitals.
        ras2 : int 
            Number of RAS2 orbitals.
        ras3 : int 
            Number of RAS3 orbitals.
        conf_space : string
            Excitations to include (hole, particle, etc).

        Returns
        -------
        TEIFullPsi4
            Initialized TEI object.
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
        self.eri = psi4.core.Matrix.to_array(
                       mints.mo_eri(C_act,C_act,C_act,C_act))
        # put in physicists' notation
        self.eri = self.eri.transpose(0, 2, 1, 3)
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        if(conf_space == ""):
            self.ind = [[0,0],[0,0],[0,ras2],[0,0]]
        if(conf_space == "h"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[0,0]]
        if(conf_space == "p"):
            self.ind = [[0,0],[0,0],[0,ras2],[ras2, ras2+ras3]]
        if(conf_space == "h,p"):
            self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],
                        [ras1+ras2,ras1+ras2+ras3]]
        print("Constructed TEI object in %i seconds." 
              %(time.time() - tei_start_time))

# DF TEI INTEGRALS

class TEIDFBase(TEI):
    """Base class for constructing density-fitted two-electron integrals.
    """
    @abstractmethod
    def __init__(self):
        """Constructs TEI DF objects (abstract).
        """
        pass

    def get_subblock(self, a, b, c, d):
        """
        Returns a given subblock of the two-electron integrals (DF).

        Returns a given subblock of the DF two-electron integral object.
        The RAS space to return is given by ``a``, ``b``, ``c``, 
        and ``d``. This function performs the contraction of the given 
        bra (B_ab) and ket (B_cd) matrices using einsum. The output 
        is given in physicists' notation with the form <ab|cd>.

        Parameters
        ----------
        a : int
            RAS space of index 1.
        b : int 
            RAS space of index 2.
        c : int
            RAS space of index 3.
        d : int
            RAS space of index 4.

        Returns
        -------
        numpy.ndarray
            NumPy representation of the desired subblock.
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
    """
    Class for constructing TEIs using NumPy arrays.

    This allows for construction of DF integrals using NumPy arrays as 
    input. B matrices are constructed in the initialization step and 
    are subsequently multiplied to form the appropriate subblocks.
    B matrices are only formed for the necessary subblocks, given the 
    excitation scheme.

    Attributes
    ----------
    B11 : numpy.ndarray
        RAS1/RAS1 B-matrix.
    B12 : numpy.ndarray
        RAS1/RAS2 B-matrix.
    B13 : numpy.ndarray
        RAS1/RAS3 B-matrix.
    B21 : numpy.ndarray
        RAS2/RAS1 B-matrix.
    B22 : numpy.ndarray
        RAS2/RAS2 B-matrix.
    B23 : numpy.ndarray
        RAS2/RAS3 B-matrix.
    B31 : numpy.ndarray
        RAS3/RAS1 B-matrix.
    B32 : numpy.ndarray
        RAS3/RAS2 B-matrix.
    B33 : numpy.ndarray
        RAS3/RAS3 B-matrix.
    """
    def __init__(self, C, ras1, ras2, ras3, conf_space, np_tei, np_J):
        """
        Initialize two-electron integral object.

        This initializes a TEI object for the given basis set using Psi4. 
        The TEI is stored as a NumPy array in physicists' notation.

        Parameters
        ----------
        C : numpy.ndarray
            MO coefficient matrix.
        ras1 : int 
            Number of RAS1 orbitals.
        ras2 : int 
            Number of RAS2 orbitals.
        ras3 : int 
            Number of RAS3 orbitals.
        conf_space : string
            Excitations to include (hole, particle, etc).
        np_tei : numpy.ndarray
            Basic NumPy two-electron integrals (AO basis).
        np_J : numpy.ndarray
            The J matrix for rotation.

        Returns
        -------
        TEIDFNumPy
            Initialized TEIDFNumPy object.
        """
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
        print("Constructed TEI object in %i seconds." 
              %(time.time() - tei_start_time))

class TEIDFPsi4(TEIDFBase):
    # Used Psi4NumPy for reference for this section
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space):
        """
        Initialize density-fitted TEI object using Psi4.

        This constructs a density-fitted TEI object using Psi4. 
        The B matrices are only constructed for the necessary subblocks 
        given the configuration space.

        Parameters
        ----------
        C : numpy.ndarray
            MO coefficient matrix.
        basis : psi4.core.BasisSet
            Psi4 basis set to use for integral construction.
        aux : psi4.core.BasisSet
            Auxiliary Psi4 basis set to use for integral construction.
        ras1 : int 
            Number of RAS1 orbitals.
        ras2 : int 
            Number of RAS2 orbitals.
        ras3 : int 
            Number of RAS3 orbitals.
        conf_space : string
            Excitations to include (hole, particle, etc).

        Returns
        -------
        TEIDFNumPy
            Initialized TEIDFPsi4 object.
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
        print("Constructed TEI object in %i seconds." 
              %(time.time() - tei_start_time))


