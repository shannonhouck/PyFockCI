import psi4
import numpy as np

def get_F(wfn):
    """Gets Fa and Fb for spatial orbitals with Psi4.
       :param wfn: Psi4 wavefunction object
       :return: Tuple with Fock matrices
    """
    # get necessary integrals/matrices from Psi4 (AO basis)
    C = psi4.core.Matrix.to_array(wfn.Ca())
    Fa = psi4.core.Matrix.to_array(wfn.Fa(), copy=True)
    Fb = psi4.core.Matrix.to_array(wfn.Fb(), copy=True)
    # transform to MO
    Fa = np.dot(C.T, np.dot(Fa, C)) 
    Fb = np.dot(C.T, np.dot(Fb, C)) 
    return (Fa, Fb) 

