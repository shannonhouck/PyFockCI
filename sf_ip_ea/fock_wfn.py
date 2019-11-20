"""
Fock CI calculation information storage.

This module contains a class for storing information about RAS-nSF-IP/EA 
calculations. This class is returned by ``fock_ci()``, allowing information 
about the wavefunction to be accessed in a Python script even after the 
calculation itself is complete. 
"""

# importing general python functionality
from __future__ import print_function

# importing numpy
import numpy as np

# importing our packages
from .post_ci_analysis import *

class FockWfn:
    """
    Contains information about the RAS-SF-IP/EA calculation.

    The FockWfn class stores important information about the calculation, 
    including RAS spaces, eigenvectors, energies, S**2 values, and so on. 
    This allows information to be accessed after the calculation is 
    complete.

    Attributes
    ----------
    n_SF : int
        Number of spin-flips to perform (SF)
    delta_ec : int
        Change in electron count (IP/EA)
    conf_space : str
        Configuration space
    ras1 : int
        RAS1 orbital count
    ras2 : int
        RAS2 orbital count
    ras3 : int
        RAS3 orbital count
    n_roots : int
        Number of roots calculated
    n_dets : int
        Number of determinants
    e : numpy.ndarray
        Calculated energy values
    vecs : numpy.ndarray
        Calculated eigenvectors
    local_vecs : numpy.ndarray
        Localized eigenvectors (if requested)
    s : numpy.ndarray
        Array of S values (per root)
    sz : numpy.ndarray
        Array of Sz values (per root)
    s2 : numpy.ndarray
        Array of S**2 values (per root)
    H : numpy.ndarray
        Calculated Hamiltonian (only if requested)
    wfn : psi4.core.Wavefunction
        Wavefunction ROHF wavefunction object. This can be replaced with 
        wavefunction-like objects for other programs (ex. PySCF's SCF class).
    """

    def __init__(self, n_SF, delta_ec, conf_space, ras1, ras2, ras3, 
                 n_roots, n_dets, det_list):
        """
        Creates and initializes a wfn_sf object.

        This initializes the wfn_sf object, which holds information about the
        calculation performed. Sets things that are known at the beginning of 
        the calculation; other things are initialized but not set until 
        the calculation is complete.

        Parameters
        ----------
        n_SF : int
            Number of spin-flips to perform (SF)
        delta_ec : int
            Change in electron count (IP/EA)
        conf_space : str
            Configuration space
        ras1 : int
            RAS1 orbital count
        ras2 : int
            RAS2 orbital count
        ras3 : int
            RAS3 orbital count
        n_roots : int
            Number of roots calculated
        n_dets : int
            Number of determinants
        det_list : list
            An ordered list of the determinants in the Fock space
        """
        # active space info
        self.n_SF = n_SF
        self.delta_ec = delta_ec
        self.conf_space = conf_space
        self.ras1 = ras1
        self.ras2 = ras2
        self.ras3 = ras3
        # eigs info
        self.e = np.zeros((n_roots))
        self.vecs = np.zeros((n_dets, n_roots))
        self.local_vecs = np.zeros((n_dets, n_roots))
        # determinant/root info
        self.n_roots = n_roots
        self.n_dets = n_dets
        self.det_list = det_list
        # spin/multiplicity info
        self.s = np.zeros((n_roots))
        self.sz = np.zeros((n_roots))
        self.s2 = np.zeros((n_roots))
        # Psi4-specific (unset otherwise)
        self.H = None
        self.wfn = None

    def print_roots(self):
        """
        Prints the calculated root energies, with S**2 and Sz values.

        This prints the energy, Sz, and S**2 values for each root to 
        standard output.
        """
        print("\nROOT No.\tEnergy\t\t\tSz\tS**2")
        print("----------------------------------------------------------")
        for i in range(self.n_roots):
            print("   %i\t\t%12.12f\t%3.3f\t%8.6f" 
                  % (i, self.e[i], self.sz[i], self.s2[i]))
        print("----------------------------------------------------------\n")

    def print_important_dets(self):
        """
        Prints the most important determinants for each root.

        This prints the most important determinants to standard output 
        for each root, with information about which orbitals had electrons 
        eliminated from or added to them.
        """
        print("Most Important Determinants Data:")
        for i, corr in enumerate(self.e):
            print("\nROOT %i: %12.12f" %(i, corr))
            print_dets(self.vecs[:,i], self)

    def print_local_dets(self):
        """
        Prints the most important determinants for each root (localized).

        This prints the most important determinants for each root to 
        standard output, using localized orbitals.
        """
        print("Most Important Determinants Data (Localized):")
        for i, corr in enumerate(self.e):
            print("\nROOT %i: %12.12f" %(i, corr))
            print_dets(self.local_vecs[:,i], self)

