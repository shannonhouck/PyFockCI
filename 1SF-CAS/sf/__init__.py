# importing general python functionality
from __future__ import print_function
import math

# importing numpy
import numpy as np

# importing our packages
from .psi4_sf import do_sf_psi4
from .np_sf import do_sf_np

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

# Performs Fock-space CI (SF-IP/EA)
#    delta_a         Desired number of alpha electrons to remove (int)
#    delta_b         Desired number of beta electrons to add (int)
#    ref_opts        Options for the reference code (dict)
#                    See relevant reference code (ex. do_sf_psi4) for details
#    sf_opts         Additional options (dict)
#                        sf_diag_method Diagonalization method to use.
#                            "RSP"        Direct (deprecated)
#                            "LANCZOS"    Use NumPy's Lanczos
#                            "DAVIDSON"   Use our Davidson
#                        num_roots      Number of roots to solve for.
#                        guess_type     Type of guess vector to use (CAS vs. RANDOM)
#                            "CAS"        Do a (usually inexpensive) CAS first and use that as an initial guess.
#                            "RANDOM"     Random orthonormal basis
#                            "READ"       Read guess from a NumPy file (not yet supported)
#                            "I"          Identity
#                        integral_type  Which type of integrals to use (DF or FULL)
#                             "FULL"      Use full integrals (no density fitting)
#                             "DF"        Use density fit integrals
#                        return_vects   Whether to return eigenvectors (defaults to false)
#
# Returns:
#    energy          List of requested roots for the system
#    vects           Eigenvectors for the system (optional)
#    wfn             Psi4 reference wavefunction (Psi4 only, optional)
def fock_ci(delta_a, delta_b, mol, conf_space="", ref_opts={}, sf_opts={}, program='PSI4'):
    # update options
    tmp_opts = {'SF_DIAG_METHOD': 'DAVIDSON',
                'NUM_ROOTS': 6,
                'GUESS_TYPE': 'CAS',
                'INTEGRAL_TYPE': 'FULL',
                'AUX_BASIS_TYPE': '',
                'RETURN_VECTS': False,
                'RETURN_WFN': False}
    for key in sf_opts:
        if(isinstance(sf_opts[key], str)):
            tmp_opts.update({key.upper(): sf_opts[key].upper()})
        else:
            tmp_opts.update({key.upper(): sf_opts[key]})
    sf_opts = tmp_opts
    # choose correct program
    if(program.upper()=='PSI4'):
        return do_sf_psi4(delta_a, delta_b, mol, conf_space, ref_opts, sf_opts)
    if(program.upper()=='READ'):
        if('FILE' in ref_opts):
            # read info from file
            ref = np.load(ref_opts['FILE'])
            extras = ref['extras']
            e_in = extras[0]
            ras1_in = int(extras[1])
            ras2_in = int(extras[2])
            ras3_in = int(extras[3])
            Ca_in = ref['Ca']
            Cb_in = ref['Cb']
            Fa_in = ref['Fa']
            Fb_in = ref['Fb']
            tei_in = ref['TEI']
            return do_sf_np(delta_a, delta_b, mol, ras1_in, ras2_in, ras3_in, Fa_in, Fb_in,
                            tei_in, e_in, conf_space, ref_opts, sf_opts)
        else:
            print("ERROR: Please give a filename in ref_opts{}. Exiting...")
            exit()
    # if program is not supported
    else:
        print("ERROR: Program %s is not yet supported. Exiting..." %(program) )
        exit()




