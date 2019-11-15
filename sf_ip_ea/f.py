"""
Handling of the Fock matrix.

This module is responsible for handling things related to the Fock matrix.
"""

import psi4
import numpy as np

def get_F(wfn):
    """
    Gets alpha and beta Fock matrices.

    Given a Psi4 Wavefunction object, this returns NumPy representations of 
    the alpha and beta Fock matrices in the spatial orbital MO basis.

    Parameters
    ----------
    wfn : psi4.core.Wavefunction
        Psi4 wavefunction object from which to obtain alpha and beta 
        Fock matrices.

    Returns
    -------    
    tuple
        A tuple (Fa, Fb) containing NumPy representations of the alpha 
        and beta Fock matrices, respectively.
    """
    # get necessary integrals/matrices from Psi4 (AO basis)
    C = psi4.core.Matrix.to_array(wfn.Ca())
    Fa = psi4.core.Matrix.to_array(wfn.Fa(), copy=True)
    Fb = psi4.core.Matrix.to_array(wfn.Fb(), copy=True)
    # transform to MO
    Fa = np.dot(C.T, np.dot(Fa, C)) 
    Fb = np.dot(C.T, np.dot(Fb, C)) 
    return (Fa, Fb) 

