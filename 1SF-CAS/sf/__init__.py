# importing general python functionality
from __future__ import print_function
import math

# importing numpy
import numpy as np
from numpy import linalg as LIN
from scipy.sparse import linalg as SPLIN
from linop import LinOpH

# importing Psi4
import psi4

# importing our packages
import f
import tei
import post_ci_analysis

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

# Runs SF-CAS using Psi4's integral packages
def sf_psi4(delta_a, delta_b, mol, conf_space="", add_opts={}, sf_diag_method="LinOp",
            num_roots=6, guess_type="CAS", integral_type="FULL", aux_basis_name="", return_vects=False, return_wfn=False):
    # cleanup in case of multiple calculations
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    # setting default options, reading in additional options from user
    opts = {'basis': 'cc-pvdz',
            'scf_type': 'direct',
            'e_convergence': 1e-10,
            'd_convergence': 1e-10,
            'reference': 'rohf'}
    opts.update(add_opts)
    psi4.set_options(opts)
    # run ROHF
    e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    # obtain some important values
    ras1 = wfn.doccpi()[0]
    ras2 = wfn.soccpi()[0]
    ras3 = wfn.basisset().nbf() - (ras1 + ras2)
    # get integrals
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    Fa, Fb = f.get_F(wfn)
    if(integral_type=="FULL"):
        tei_int = tei.TEIFull(wfn.Ca(), wfn.basisset(), wfn.doccpi()[0], ras2, ras3)
    if(integral_type=="DF"):
        # if user hasn't defined which aux basis to use, default behavior is to use the one from opts
        if(aux_basis_name == ""):
            aux_basis_name = opts['basis']
        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", aux_basis_name)
        tei_int = tei.TEIDF(wfn.Ca(), wfn.basisset(), aux_basis, ras1, ras2, ras3, conf_space)
    out = do_sf_cas(delta_a, delta_b, mol, ras1, ras2, ras3, Fa, Fb, tei_int, e, conf_space, sf_diag_method, num_roots,
                    guess_type, integral_type, aux_basis_name, return_vects)
    # return appropriate values
    if(return_wfn):
        return (out,) + (wfn,)
    else:
        return out


# Performs the 1SF-CAS calculation.
# Parameters:
#    delta_a         Desired number of alpha electrons to remove
#    delta_b         Desired number of beta electrons to add
#    mol             Molecule to run calculation on
#    conf_space      Desired excitation scheme:
#                        ""       1SF-CAS
#                        "h"      1SF-CAS + h
#                        "p"      1SF-CAS + p
#                        "h,p"    1SF-CAS + (h,p)
#    add_opts        Additional options (to be passed on to Psi4)
#    sf_diag_method  Diagonalization method to use.
#                        "RSP"    Direct
#                        "LinOp"  LinOp
#    num_roots       Number of roots to solve for.
#    guess_type      Type of guess vector to use (CAS vs. RANDOM)
#
# Returns:
#    energy          Lowest root found by eigensolver (energy of system)
def do_sf_cas(delta_a, delta_b, mol, ras1, ras2, ras3, Fa, Fb, tei_int, e, conf_space="",
              sf_diag_method="LinOp", num_roots=6, guess_type="CAS", integral_type="FULL", aux_basis_name="", return_vects=False ):
    # determine number of spin-flips and total change in electron count
    delta_ec = delta_b - delta_a
    n_SF = min(delta_a, delta_b)
    # determine number of determinants
    # RAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        n_dets = ras2 * ras2
    elif(n_SF==2 and delta_ec==0 and conf_space==""):
        guess_type = ""
        n_dets = 0
        for i in range(ras2):
            for j in range(i):
                for a in range(ras2):
                    for b in range(a):
                        n_dets = n_dets + 1 
    elif(n_SF==1 and delta_ec==0 and conf_space=="h"):
        n_dets = (ras2 * ras2) + (ras2 * ras1) + (((ras2-1)*(ras2)/2) * ras2 * ras1)
    elif(n_SF==1 and delta_ec==0 and conf_space=="p"):
        n_dets = (ras2 * ras2) + (ras2 * ras3) + (((ras2-1)*(ras2)/2) * ras2 * ras3)
    elif(n_SF==1 and delta_ec==0 and (conf_space=="h,p" or conf_space=="1x")):
        n_dets = (ras2 * ras2) + (ras2 * ras1) + (((ras2-1)*(ras2)/2) * ras2 * ras1)
        n_dets = n_dets + (ras2 * ras3) + (((ras2-1)*(ras2)/2) * ras2 * ras3)
    # CAS-IP/EA
    elif(n_SF==0 and (delta_ec==-1 or delta_ec==1) and conf_space==""):
        guess_type = ""
        n_dets = ras2
    # RAS(h)-EA
    elif(n_SF==0 and delta_ec==1 and conf_space=="h"):
        guess_type = ""
        n_dets = ras2
        for I in range(ras1):
            for a in range(ras2):
                for b in range(a):
                    n_dets = n_dets + 1
    # RAS(p)-EA
    elif(n_SF==0 and delta_ec==1 and conf_space=="p"):
        guess_type = ""
        n_dets = ras2 + ras3 + (ras3*ras2*ras2)
    # RAS(h)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="h"):
        guess_type = ""
        n_dets = ras2 + ras1 + (ras1*ras2*ras2)
    # RAS(p)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="p"):
        guess_type = ""
        n_dets = ras2
        for i in range(ras2):
            for j in range(i):
                for A in range(ras3):
                    n_dets = n_dets + 1
    # CAS-1SF-IP/EA
    elif(n_SF==1 and (delta_ec==-1 or delta_ec==1) and conf_space==""):
        guess_type = ""
        n_dets = ras2 * ((ras2-1)*(ras2)/2)
    # RAS(p)-1SF-EA
    elif(n_SF==1 and delta_ec==1 and conf_space=="p"):
            n_dets = ras2 * ras3 * ras2
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        n_dets = n_dets + 1
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for a in range(ras2):
                            for b in range(a):
                                n_dets = n_dets + 1
    # RAS(h)-1SF-IP
    elif(n_SF==1 and delta_ec==-1 and conf_space=="h"):
        guess_type = ""
        n_dets = ras2 * ((ras2-1)*(ras2)/2)
        n_dets = n_dets + (ras2 * ras1 * ras2)
        # this is the MOST hack-ish way to get the triangle number of a traingle number. Fix later
        count = 0 
        for I in range(ras1):
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        for b in range(a):
                            count = count + 1
        n_dets = n_dets + count
    else:
        print("Sorry, %iSF with electron count change of %i not yet supported. Exiting..." %(n_SF, delta_ec) )
        exit()
    # make sure n_dets is an int (for newer Python versions)
    n_dets = int(n_dets)
    #if(sf_diag_method == "RSP"):
    #    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0:8])
    #    print("FROM DIAG: ", np.sort(LIN.eigvalsh(H))[0:6])
    #    return(e + np.sort(LIN.eigvalsh(H))[0])
    print("Performing %iSF with electron count change of %i..." %(n_SF, delta_ec) )
    print("\tRAS1: %i\n\tRAS2: %i\n\tRAS3: %i" %(ras1, ras2, ras3) )
    if(sf_diag_method == "LinOp"):
        #A = SPLIN.LinearOperator(H.shape, matvec=mv)
        a_occ = ras1 + ras2
        b_occ = ras1
        a_virt = ras3
        b_virt = ras2 + ras3
        print("Number of determinants:", n_dets)
        if( num_roots >= n_dets ):
            num_roots = n_dets - 1
        if(conf_space==""):
            A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in=conf_space)
            vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots)
        else:
            if("guess_type"=="CAS"):
                cas_A = LinOpH((ras2*ras2,ras2*ras2), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in="")
                cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA', k=1)
                socc = wfn.soccpi()[0]
                v3_guess = np.zeros((n_dets-(ras2*ras2), 1))
                guess_vect = np.vstack((cas_vects, v3_guess)).T
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA', v0=guess_vect)
            else:
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei_int, n_SF, delta_ec, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots)
        print("\nROOT No.\tEnergy\t\tS**2")
        print("------------------------------------------------")
        for i, corr in enumerate(vals):
            s2 = post_ci_analysis.calc_s_squared(n_SF, delta_ec, conf_space, vects[:, i], ras1, ras2, ras3)
            print("   %i\t\t%6.6f\t%8.6f" % (i, e + corr, s2))
        print("------------------------------------------------\n")
    # Return appropriate things
    if(return_vects):
        return (e + vals, vects)
    else:
        return e + vals


