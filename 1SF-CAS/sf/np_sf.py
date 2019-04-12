# importing general python functionality
from __future__ import print_function
import math

# importing numpy
import numpy as np
from numpy import linalg as LIN
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
RAS-SF-IP/EA PROGRAM

Runs RAS-SF-IP/EA calculations. In Progress.

Refs:
Crawford Tutorials (http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12)
DePrince Tutorials (https://www.chem.fsu.edu/~deprince/programming_projects/cis/)
Sherrill Notes (http://vergil.chemistry.gatech.edu/notes/cis/cis.html)
Psi4NumPy Tutorials
"""

###############################################################################################
# Functions
###############################################################################################

# Performs the 1SF-CAS calculation, given NumPy input arrays.
# Parameters:
#    delta_a         Desired number of alpha electrons to remove (int)
#    delta_b         Desired number of beta electrons to add (int)
#    ras1            RAS1 space (int)
#    ras2            RAS2 space (int)
#    ras3            RAS3 space (int)
#    Fa              Alpha Fock matrix (NumPy array)
#    Fb              Beta Fock matrix (NumPy array)
#    tei_int         Two-electron integrals (NumPy array)
#    e               ROHF energy (float)
#    conf_space      Desired excitation scheme:
#                        ""         CAS-nSF-IP/EA (default)
#                        "h"        RAS(h)-nSF-IP/EA
#                        "p"        RAS(p)-nSF-IP/EA
#                        "h,p"      RAS(h,p)-nSF-IP/EA
#    sf_opts         Additional options (dict)
#                    See __init__ for more information.
#
# Returns:
#    energy          List of requested roots for the system
#    vects           Eigenvectors for the system (optional)
def do_sf_np(delta_a, delta_b, ras1, ras2, ras3, Fa, Fb, tei_int, e, conf_space="", sf_opts={}, J_in=None, C_in=None):
    # update options
    opts =  {'SF_DIAG_METHOD': 'DAVIDSON',
             'NUM_ROOTS': 6,
             'GUESS_TYPE': 'CAS',
             'INTEGRAL_TYPE': 'FULL',
             'AUX_BASIS_TYPE': '',
             'RETURN_VECTS': False,
             'RETURN_WFN': False}
    for key in sf_opts:
        if(isinstance(sf_opts[key], str)):
            opts.update({key.upper(): sf_opts[key].upper()})
        else:
            opts.update({key.upper(): sf_opts[key]})
    # make TEI object if we've passed in a numpy array
    print(opts)
    if(opts['INTEGRAL_TYPE']=="FULL"):
        tei_int = TEIFull(0, 0, ras1, ras2, ras3, np_tei=tei_int)
    elif(opts['INTEGRAL_TYPE']=="DF"):
        tei_int = TEIDF(C_in, 0, 0, ras1, ras2, ras3, conf_space, np_tei=tei_int, np_J=J_in)
    # determine number of spin-flips and total change in electron count
    n_SF = min(delta_a, delta_b)
    delta_ec = delta_b - delta_a
    # generate list of determinants (for counting and other purposes)
    print(opts)
    print(conf_space)
    det_list = generate_dets(n_SF, delta_ec, conf_space, ras1, ras2, ras3)
    n_dets = len(det_list)
    # make sure n_dets is an int (for newer Python versions)
    n_dets = int(n_dets)
    # setup for method
    print("Performing %iSF with electron count change of %i..." %(n_SF, delta_ec) )
    print("\tRAS1: %i\n\tRAS2: %i\n\tRAS3: %i" %(ras1, ras2, ras3) )
    print("Number of determinants:", n_dets)
    # make sure we're only solving for an appropriate number of roots
    # should make RSP later for small Hamiltonian sizes
    num_roots = opts['NUM_ROOTS']
    if( num_roots >= n_dets ):
        num_roots = n_dets - 1
    num_roots = int(num_roots)
    print("Number of roots:", num_roots)
    a_occ = ras1 + ras2
    b_occ = ras1
    a_virt = ras3
    b_virt = ras2 + ras3
    # Generate appropriate guesses
    guess_vect = None
    A = LinOpH((n_dets,n_dets), e, a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in=conf_space)
    # run method
    print("Running Fock-space CI...")
    print("\tSpin-Flips: %3i\n\tElectron Count Change: %3i\n" %(n_SF, delta_ec))
    print("\tRAS1: %i\n\tRAS2: %i\n\tRAS3: %i" %(ras1, ras2, ras3) )
    if(n_dets < 250):
        opts['SF_DIAG_METHOD'] = "LANCZOS"
    print("\tDiagonalization: %s\n\tGuess: %s" %(opts['SF_DIAG_METHOD'], opts['GUESS_TYPE']))
    if(opts['SF_DIAG_METHOD'] == "LANCZOS"):
        # generate guess vector
        #if(guess_type == "CAS"):
        #    if(conf_space==""):
        #        guess_vect = LIN.orth(np.random.rand(n_dets, num_roots))
        #        print(guess_vect.shape)
        #    else:
        #        cas_A = linop.LinOpH((ras2*ras2,ras2*ras2), e, a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in="")
        #        cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA', k=1)
        #        v3_guess_padding = np.zeros((n_dets-(ras2*ras2), 1)) 
        #        guess_vect = np.vstack((cas_vects, v3_guess_padding)).T
        # do LANCZOS
        if(conf_space==""):
            vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA')
        else:
            #vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA', v0=guess_vect)
            vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA')
    else: #if(sf_diag_method == "DAVIDSON"):
        # generate guess vector
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
                # TODO: Modify CAS root number if needed
                cas_A = linop.LinOpH((n_cas_dets,n_cas_dets), e, a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in="")
                cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA', k=num_roots)
                v3_guess = np.zeros((n_dets-(n_cas_dets), num_roots)) 
                guess_vect = np.vstack((cas_vects, v3_guess))
        elif(opts['GUESS_TYPE'] == "RANDOM"):
            guess_vect = LIN.orth(np.random.rand(n_dets, num_roots))
        else:
            guess_vect = np.zeros((n_dets, num_roots))
            for i in range(num_roots):
                guess_vect[i,i] = 1.0 
        vals, vects = davidson(A, guess_vect)
    print("\nROOT No.\tEnergy\t\tS**2")
    print("------------------------------------------------")
    for i, corr in enumerate(vals):
        s2 = calc_s_squared(n_SF, delta_ec, conf_space, vects[:, i], ras1, ras2, ras3)
        print("   %i\t\t%6.6f\t%8.6f" % (i, corr, s2))
    print("------------------------------------------------\n")
    print("Most Important Determinants Data:")
    for i, corr in enumerate(vals):
        print("\nROOT %i: %12.12f" %(i, corr))
        s2 = print_dets(vects[:,i], n_SF, delta_ec, conf_space, n_dets, ras1, ras2, ras3)
    print("\n\n\t  Fock Space CI Complete! \n")
    # Return appropriate things
    if(return_vects):
        return (vals, vects)
    else:
        return vals


