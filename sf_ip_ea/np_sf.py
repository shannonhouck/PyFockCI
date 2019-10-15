# importing general python functionality
from __future__ import print_function
import math
import time

# importing numpy
import numpy as np
from numpy import linalg as LIN
from scipy import linalg as SCILIN
from scipy.sparse import linalg as SPLIN

# importing Psi4
import psi4

# importing our packages
from .linop import LinOpH
from .f import *
from .tei import *
from .post_ci_analysis import *
from .solvers import *

"""
NumPy RAS-SF-IP/EA

Performs Fock-space CI using NumPy arrays as input for integrals.

"""

class wfn_sf:
    def __init__(self, n_SF, delta_ec, conf_space, ras1, ras2, ras3, n_roots, n_dets):
        # active space info
        self.n_SF = n_SF
        self.delta_ec = delta_ec
        self.conf_space = conf_space
        self.ras1 = ras1
        self.ras2 = ras2
        self.ras3 = ras3
        self.n_roots = n_roots
        self.n_dets = n_dets
        # eigs info
        self.e = np.zeros((n_roots))
        self.vecs = np.zeros((n_dets, n_roots))
        self.local_vecs = np.zeros((n_dets, n_roots))
        # spin/multiplicity info
        self.s = np.zeros((n_roots))
        self.sz = np.zeros((n_roots))
        self.s2 = np.zeros((n_roots))
        # Psi4-specific (unset otherwise)
        self.wfn = None

    def print_roots(self):
        print("\nROOT No.\tEnergy\t\t\tSz\tS**2")
        print("----------------------------------------------------------")
        for i in range(self.n_roots):
            print("   %i\t\t%12.12f\t%3.3f\t%8.6f" % (i, self.e[i], self.sz[i], self.s2[i]))
        print("----------------------------------------------------------\n")

    def print_important_dets(self):
        print("Most Important Determinants Data:")
        for i, corr in enumerate(self.e):
            print("\nROOT %i: %12.12f" %(i, corr))
            print_dets(self.vecs[:,i], self.n_SF, self.delta_ec, self.conf_space, self.ras1,
                       self.ras2, self.ras3)

    def print_local_dets(self):
        print("Most Important Determinants Data (Localized):")
        for i, corr in enumerate(self.e):
            print("\nROOT %i: %12.12f" %(i, corr))
            print_dets(self.local_vecs[:,i], self.n_SF, self.delta_ec, self.conf_space, self.ras1,
                       self.ras2, self.ras3)

def do_sf_np(delta_a, delta_b, ras1, ras2, ras3, Fa, Fb, tei_int, e,
             conf_space="", sf_opts={}, J_in=None, C_in=None):
    """Performs the 1SF-CAS calculation, given NumPy input arrays.
       Input
           delta_a -- Desired number of alpha electrons to remove (int)
           delta_b -- Desired number of beta electrons to add (int)
           ras1 -- RAS1 space (int)
           ras2 -- RAS2 space (int)
           ras3 -- RAS3 space (int)
           Fa -- Alpha Fock matrix (NumPy array)
           Fb -- Beta Fock matrix (NumPy array)
           tei_int -- Two-electron integrals (NumPy array)
           e -- ROHF energy (float)
           conf_space -- Desired excitation scheme:
                         "" CAS-nSF-IP/EA (default)
                         "h" RAS(h)-nSF-IP/EA
                         "p" RAS(p)-nSF-IP/EA
                         "h,p" RAS(h,p)-nSF-IP/EA
           sf_opts -- Additional options (dict)
                      See __init__ for more information.
           J_in -- J matrix (for DF calculations)
           C_in -- MO coefficients matrix (for DF calculations)
       Output
           energy -- List of requested roots for the system
           vects -- Eigenvectors for the system (optional)
    """
    # OPTIONS HANDLING
    opts =  {'SF_DIAG_METHOD': 'DAVIDSON',
             'NUM_ROOTS': 6,
             'GUESS_TYPE': 'CAS',
             'INTEGRAL_TYPE': 'FULL',
             'AUX_BASIS_TYPE': ''}
    # capitalize as needed
    for key in sf_opts:
        if(isinstance(sf_opts[key], str)):
            opts.update({key.upper(): sf_opts[key].upper()})
        else:
            opts.update({key.upper(): sf_opts[key]})
    # INTEGRAL HANDLING
    # make TEI object if we've passed in a numpy array
    if(isinstance(tei_int, np.ndarray)):
        if(opts['INTEGRAL_TYPE']=="FULL"):
            tei_int = TEIFull(0, 0, ras1, ras2, ras3, np_tei=tei_int)
        elif(opts['INTEGRAL_TYPE']=="DF"):
            tei_int = TEIDF(C_in, 0, 0, ras1, ras2, ras3, conf_space,
                            np_tei=tei_int, np_J=J_in)
    # CALCULATION SETUP
    # determine number of spin-flips and total change in electron count
    n_SF = min(delta_a, delta_b)
    delta_ec = delta_b - delta_a
    # generate list of determinants (for counting and other purposes)
    det_list = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)
    n_dets = int(len(det_list))
    # make sure we're only solving for an appropriate number of roots
    # TODO: make RSP option for small Hamiltonian sizes
    num_roots = opts['NUM_ROOTS']
    if( num_roots >= n_dets ):
        num_roots = n_dets
    num_roots = int(num_roots)
    # set up wfn (for returning)
    wfn = wfn_sf(n_SF, delta_ec, conf_space, ras1, ras2, ras3, num_roots, n_dets)
    # set up LinOp for diagaonalization
    A = LinOpH((n_dets,n_dets), e, ras1, ras2, ras3, Fa, Fb, tei_int, n_SF,
               delta_ec, conf_space_in=conf_space)
    # PRINT INFO
    # print info about calculation
    print("Setup complete! Performing %iSF with electron count change of %i..."
          %(n_SF, delta_ec) )
    print("\tRAS1: %i\n\tRAS2: %i\n\tRAS3: %i" %(ras1, ras2, ras3) )
    print("Number of Roots:", num_roots)
    print("Number of Determinants:", n_dets)
    print("Configuration Space:", conf_space)
    print("Additional Options:", opts)
    print("\tSpin-Flips: %3i\n\tElectron Count Change: %3i\n"
          %(n_SF, delta_ec))
    print("\tRAS1: %i\n\tRAS2: %i\n\tRAS3: %i" %(ras1, ras2, ras3) )
    if(n_dets < 250):
        opts['SF_DIAG_METHOD'] = "LANCZOS"
    print("\tDiagonalization: %s\n\tGuess: %s" %(opts['SF_DIAG_METHOD'],
                                                 opts['GUESS_TYPE']))
    # DIAGONALIZATION
    # use built-in Lanczos method
    # TODO: Support other guess options
    start_diag_time = time.time()
    if(opts['SF_DIAG_METHOD'] == "LANCZOS"):
        # do LANCZOS
        if(num_roots == n_dets):
            H_full = A.matmat(np.eye(n_dets))
            vals, vects = SCILIN.eigh(H_full)
        else:
            vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA')
    # use Davidson method
    elif(opts['SF_DIAG_METHOD'] == "DAVIDSON"):
        # generate guess vector
        guess_vect = None
        # CAS guess
        if(opts['GUESS_TYPE'] == "CAS"):
            if(conf_space==""):
                guess_vect = LIN.orth(np.random.rand(n_dets, num_roots))
            else:
                # CAS-1SF
                if(n_SF==1 and delta_ec==0):
                    n_cas_dets = int(ras2 * ras2)
                # CAS-1SF-IP/EA
                elif(n_SF==1 and (delta_ec==-1 or delta_ec==1)):
                    n_cas_dets = int(ras2 * ((ras2-1)*(ras2)/2))
                # CAS-IP/EA
                elif(n_SF==0 and (delta_ec==-1 or delta_ec==1)):
                    n_cas_dets = int(ras2)
                # Modify CAS root number if needed
                cas_A = LinOpH((n_cas_dets,n_cas_dets), e, ras1, ras2, ras3,
                                Fa, Fb, tei_int, n_SF, delta_ec,
                                conf_space_in="")
                if(num_roots > n_cas_dets):
                    print("Number of roots requested is greater than number \
                           of CAS determinants. Using RANDOM guess instead.")
                    guess_vect = LIN.orth(np.random.rand(n_dets, num_roots))
                elif(num_roots == n_cas_dets):
                    H_full = cas_A.matmat(np.eye(n_cas_dets))
                    cas_vals, cas_vects = SCILIN.eigh(H_full)
                    v3_guess = np.zeros((n_dets-(n_cas_dets), num_roots)) 
                    print(cas_vals)
                    guess_vect = np.vstack((cas_vects, v3_guess))
                else:
                    cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA',
                                                  k=num_roots)
                    v3_guess = np.zeros((n_dets-(n_cas_dets), num_roots)) 
                    guess_vect = np.vstack((cas_vects, v3_guess))
        # random guess vector
        elif(opts['GUESS_TYPE'] == "RANDOM"):
            guess_vect = LIN.orth(np.random.rand(n_dets, num_roots))
        # otherwise, just use identity
        else:
            guess_vect = np.zeros((n_dets, num_roots))
            for i in range(num_roots):
                guess_vect[i,i] = 1.0 
        # do Davidson
        vals, vects = davidson(A, guess_vect)
    else:
        print("Diag method not yet supported. \
               Please use DAVIDSON or LANCZOS.")
        exit()
    print("Diagonalization completed in %i seconds." %(time.time() - start_diag_time) )

    # POST-HF ANALYSIS
    # set eigenvalues/vects
    wfn.e = vals
    wfn.vecs = vects
    # set spin/mult results
    for i, corr in enumerate(vals):
        wfn.sz[i] = calc_sz(n_SF, delta_ec, conf_space, vects[:, i],
                            ras1, ras2, ras3)
        wfn.s2[i] = calc_s2(n_SF, delta_ec, conf_space, vects[:, i],
                            ras1, ras2, ras3)
        wfn.s[i] = 0.5*(math.sqrt(1.0+4.0*wfn.s2[i]))-0.5

    # print info about determinants and coefficients
    wfn.print_roots()
    wfn.print_important_dets()
    
    return wfn


