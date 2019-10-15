# importing general python functionality
from __future__ import print_function
import time
import math

# importing necessary packages
import numpy as np
import psi4

# importing our packages
from .f import get_F
from .tei import *
from .np_sf import do_sf_np

def do_sf_psi4(delta_a, delta_b, mol, conf_space="", ref_opts={}, sf_opts={}):
    """Runs SF-CAS using Psi4's integral packages
       Input
           delta_a -- Number of alpha electrons to eliminate
           delta_b -- Number of beta electrons to create
           mol -- Psi4 Molecule object on which to run the calculation
           conf_space -- Excitation scheme (Options: "", "h", "p", "h,p")
           ref_opts -- Additional options for Psi4 (dictionary)
           sf_opts -- Additional options for Fock CI (dictionary)
       Returns
           e -- List of SF-IP/EA roots (as many as requested)
           vects (opt) -- List of eigenvectors corresponding to the roots in e
           wfn (opt) -- The reference ROHF wavefunction object from Psi4
    """
    # cleanup in case of multiple calculations
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    # setting default options, reading in additional options from user
    psi4_opts = {'BASIS': 'cc-pvdz',
                 'scf_type': 'direct',
                 'e_convergence': 1e-10,
                 'd_convergence': 1e-10,
                 'guess': 'gwh',
                 'reference': 'rohf'}
    psi4_opts.update(ref_opts)
    psi4.set_options(psi4_opts)
    # run ROHF
    if(sf_opts['READ_PSI4_WFN']):
        wfn = sf_opts['PSI4_WFN']
        e = wfn.energy()
    else:
        e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    # obtain RAS spaces
    ras1 = wfn.doccpi()[0]
    ras2 = wfn.soccpi()[0]
    ras3 = wfn.basisset().nbf() - (ras1 + ras2)
    # get Fock and MO coefficients
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    Fa, Fb = get_F(wfn)
    # get two-electron integrals
    if(sf_opts['INTEGRAL_TYPE']=="FULL"):
        tei_int = TEIFull(wfn.Ca(), wfn.basisset(), ras1, ras2, ras3,
                          ref_method='PSI4')
    if(sf_opts['INTEGRAL_TYPE']=="DF"):
        # if user hasn't defined which aux basis to use, default behavior
        # is to use the one from Psi4 wfn
        if(sf_opts['AUX_BASIS_NAME'] == ""):
            aux_basis_name = wfn.basisset().name()
        else:
            aux_basis_name = sf_opts['AUX_BASIS_NAME']
        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", 
                                             "JKFIT", aux_basis_name)
        tei_int = TEIDF(wfn.Ca(), wfn.basisset(), aux_basis, ras1, ras2,
                        ras3, conf_space, ref_method='PSI4')
    out = do_sf_np(delta_a, delta_b, ras1, ras2, ras3, Fa, Fb, tei_int, e,
                   conf_space=conf_space, sf_opts=sf_opts)
    # write Psi4 wavefunction to output object
    out.wfn = wfn
    return out

