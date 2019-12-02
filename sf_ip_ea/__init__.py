"""
A program for running RAS-SF-IP/EA calculations.

This program runs RAS-SF-IP/EA calculations using an efficient 
tensor-contraction-based scheme. The contractions have been hand-derived and 
use NumPy's einsum for efficiency. The program is run primarily through 
the main ``fock_ci`` function.
"""

# importing general python functionality
from __future__ import print_function
import math

# importing numpy
import numpy as np

# importing our packages
from .bloch import do_bloch
from .psi4_sf import do_sf_psi4
from .np_sf import do_sf_np

def fock_ci(delta_a, delta_b, mol, conf_space="", ref_opts={}, sf_opts={},
            program='PSI4'):
    """Performs Fock-space CI (SF-IP/EA).

    This is the main function call for the program. Given the changes in 
    alpha and beta electron counts, it performs the correct number of 
    spin-flips and IP/EAs.

    Parameters
    ----------
    delta_a : int
        Desired number of alpha electrons to remove.
    delta_b : int
        Desired number of beta electrons to add.
    mol : Molecule
        The molecule object to run the calculation on. 
        This should be built in whichever program you'll use 
        to run the reference, and should be handled properly by the 
        reference program.
    conf_space : string
        Desired configuration space/additional excitations.
            * ``""`` CAS
            * ``"h"`` 1 hole excitation
            * ``"p"`` 1 particle excitation
            * ``"h,p"`` 1 hole + 1 particle excitation
    ref_opts : dict
        Options for the reference program.
        See relevant reference program docs (ex. Psi4 docs) for details.
    sf_opts : dict
        Additional options for stand-alone SF code. 
            * ``sf_diag_method``: Diagonalization method to use.
                * ``RSP`` Direct (deprecated)
                * ``LANCZOS`` Use NumPy's Lanczos
                * ``DAVIDSON`` Use our Davidson
            * ``num_roots``: Number of roots to solve for.
            * ``guess_type``: Type of guess vector to use
                * ``CAS`` Do CAS first and use that as an initial guess.
                * ``RANDOM`` Random orthonormal basis
                * ``READ`` Read guess from a NumPy file (TODO)
            * ``integral_type``: Which integrals to use (DF or FULL)
                * ``FULL`` Use full integrals (no density fitting)
                * ``DF`` Use density fit integrals

    Returns
    -------
    sf_wfn
        Wavefunction object (sf_wfn) containing calculation data and results
    """

    # update options to pass into SF code
    tmp_opts = {'SF_DIAG_METHOD': 'DAVIDSON',
                'READ_PSI4_WFN': False,
                'PSI4_WFN': None,
                'NUM_ROOTS': 6,
                'GUESS_TYPE': 'CAS',
                'INTEGRAL_TYPE': 'FULL',
                'AUX_BASIS_NAME': '',
                'DF_ROLE': 'JKFIT',
                'RETURN_VECTS': False,
                'RETURN_WFN': False}
    # make sure they're all caps!
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
            return do_sf_np(delta_a, delta_b, mol, ras1_in, ras2_in, ras3_in,
                            Fa_in, Fb_in, tei_in, e_in, conf_space, ref_opts,
                            sf_opts)
        else:
            print("ERROR: Please give a filename in ref_opts{}. Exiting...")
            exit()
    # if program is not supported
    else:
        print("ERROR: Program %s is not yet supported. Exiting..." %(program) )
        exit()

