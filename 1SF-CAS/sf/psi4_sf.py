# importing general python functionality
from __future__ import print_function
import math

# importing necessary packages
import numpy as np
import psi4

# importing our packages
from .f import get_F
from .tei import *
from .np_sf import do_sf_np

# Runs SF-CAS using Psi4's integral packages
# Parameters:
#    delta_a         Number of alpha electrons to eliminate
#    delta_b         Number of beta electrons to create
#    conf_space      Excitation scheme (Options: "", "h", "p", "h,p")
#    psi4_opts       Additional options for Psi4 (dictionary)
#                        See Psi4 for details.
#    sf_opts         Additional options for Fock CI (dictionary)
#                        sf_diag_method  Diagonalization method to use for SF
#                        num_roots       The number of roots to find (defaults to 6)
#                        guess_type      Initial guess to use (Options: "RANDOM", "CAS")
#                        integral_type   Form of TEIs to use (Options: "FULL", "DF")
#                        aux_basis_type  Auxiliary basis to use for DF TEIs
#                                        (Defaults to JKFIT version of given Psi4 basis)
#                        return_vects    Return CI vectors? (Defaults to False)
#                        return_wfn      Return Psi4 reference ROHF wavefunction object? (Defaults to False)
# Returns:
#    e               List of SF-IP/EA roots (as many as requested)
#    vects (opt)     List of eigenvectors corresponding to the roots in e (Defaults to False)
#    wfn (opt)       The reference ROHF wavefunction object from Psi4 (Defaults to False)
def do_sf_psi4(delta_a, delta_b, mol, conf_space="", ref_opts={}, sf_opts={}):
    # cleanup in case of multiple calculations
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    # setting default options, reading in additional options from user
    psi4_opts = {'BASIS': 'cc-pvdz',
                 'scf_type': 'direct',
                 'e_convergence': 1e-10,
                 'd_convergence': 1e-10,
                 'reference': 'rohf'}
    psi4_opts.update(ref_opts)
    psi4.set_options(psi4_opts)
    # run ROHF
    e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    # obtain some important values
    ras1 = wfn.doccpi()[0]
    ras2 = wfn.soccpi()[0]
    ras3 = wfn.basisset().nbf() - (ras1 + ras2)
    # get integrals
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    Fa, Fb = get_F(wfn)
    if(sf_opts['INTEGRAL_TYPE']=="FULL"):
        tei_int = TEIFull(wfn.Ca(), wfn.basisset(), ras1, ras2, ras3)
    if(sf_opts['INTEGRAL_TYPE']=="DF"):
        # if user hasn't defined which aux basis to use, default behavior is to use the one from opts
        if(sf_opts['AUX_BASIS_NAME'] == ""):
            aux_basis_name = ref_opts['BASIS']
        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", aux_basis_name)
        tei_int = TEIDF(wfn.Ca(), wfn.basisset(), aux_basis, ras1, ras2, ras3, conf_space)
    out = do_sf_np(delta_a, delta_b, ras1, ras2, ras3, Fa, Fb, tei_int, e, conf_space=conf_space, sf_opts=sf_opts)
    # return appropriate values
    if(isinstance(out, tuple)):
        if(sf_opts['RETURN_WFN']):
            out = out + (wfn,)
    elif(sf_opts['RETURN_WFN']):
        out = (out,) + (wfn,)
    return out

