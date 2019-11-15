"""
Davidson solver module.

This module holds the Davidson solver and related functions.
"""

from __future__ import print_function
import numpy as np
import math
from scipy import linalg as LIN
from scipy.sparse import diags
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.linalg import spsolve

def davidson( A, vInit, e_conv=1e-8, r_conv=1e-4, vect_cutoff=1e-5,
              maxIter=200, collapse_per_root=20 ):
    """
    Solves for the eigenvalues and eigenvectors of a Hamiltonian.

    Uses the Davidson method to solve for the eigenvalues and eigenvectors
    of a given Hermitian matrix A.

    Parameters
    ----------
    A : sf_ip_ea.linop.LinOpH
        LinOp object defining our matrix-vector multiply.
    vInit : numpy.ndarray
        Initial guess vector(s).
    e_conv : float
        Cutoff for energy convergence. (Default: 1e-8)
    r_conv : float
        Cutoff for residual squared convergence. (Default: 1e-4)
    vect_cutoff : float
        Cutoff for adding vectors to Krylov space. (Default: 1e-5)
    maxIter : int
        Maximum number of iterations. (Default: 200)
    collapse_per_root : int
        Number of vectors per root to save after collapsing the Krylov space.
        (Default: 20)

    Returns 
    -------
    numpy.ndarray
        Eigenvalues/eigenvectors for the Hamiltonian.
    """
    # initialize vSpace (search subspace)
    # NOTE: Rows are determinants, cols are n_roots
    vSpace = vInit
    # number of eigenvalues to solve for
    k = vInit.shape[1]
    # iterations completed
    j = 0
    # storing sigmas...
    sig = None;
    # diagonal of Hamiltonian
    D = A.diag()
    # storing last eigenvalues
    lastVals = np.zeros((k))

    collapseSize = k*collapse_per_root

    # index of last sigma added
    lastSig = 0

    # keep track of converged roots
    converged = k*[False]
    final_vects = k*[None]
    final_vals = k*[0.0]

    print("Starting Davidson...")
    while ( j<maxIter ):
        # first iteration
        if(type(sig)==type(None)):
            # form k sigma vectors
            sig = A.matmat(vSpace) 
            """
            if(add_s_squared):
                sig = sig + calc_s_squared_vectors(vSpace)
            for i in range(0, vSpace.shape[1]):
                if(type(sig)==type(None)):
                    sig = A.matvec(vSpace[:,i])
                    sig = sig.reshape((vSpace.shape[0],1))
                else:
                    sig = np.column_stack((sig, A.matvec(vSpace[:,i])))
            """

        # form k sigma vectors
        else:
            sig = np.column_stack((sig, A.matmat(vSpace[:, sig.shape[1]:])))
  
        # form subspace matrix
        Av = np.dot(vSpace.T, sig)
        print(Av.shape)
        # solve for k lowest eigenvalues/vectors
        eVals, eVects = LIN.eigh(Av)
        eVals = eVals[:k]
        eVects = eVects[:, :k]

        r = None;
        # compute residuals
        for i in range(k):
            rNew = (np.dot(sig, eVects[:,i]) 
                   - eVals[i]*np.dot(vSpace, eVects[:,i]))
            if(type(r)==type(None)):
                r = rNew
            else:
                r = np.column_stack((r, rNew));

        # print info about this iteration
        print("\nIteration: %4i" % j)
        for i, val in enumerate(eVals):
            print("\tROOT %2i: %16.8f\t%E\t%E" 
                  % (i, val, eVals[i]-lastVals[i], LIN.norm(r[:,i])))

        # check residuals for convergence
        all_converged = True
        for i in range(k):
            # check residual
            if( abs(LIN.norm(r[:,i]))>r_conv ):
                all_converged = False
            # check energy
            if( abs(eVals[i]-lastVals[i]) > e_conv):
                all_converged = False
            '''
            # if one of the roots has converged for the first time...
            if( (abs(LIN.norm(r[:,i])) < r_conv) and 
                (abs(eVals[i]-lastVals[i]) < e_conv)):
                if(not converged[i]):
                    converged[i] = True
                    tmp = vSpace[:, -k:]
                    print(tmp.shape)
                    final_vects[i] = tmp[:, i]
                    final_vals[i] = eVals[i]
            '''

        # if converged, return appropriate values
        if ( all_converged ):
            print("\nDavidson Converged! \n")
            return (eVals, np.dot(vSpace, eVects))
            #return (eVals, eVects)
            #return (eVals, vSpace[-k:,:].T)
            #return (final_vals, final_vects_out)
        # collapse subspace if necessary (if)
        elif( vSpace.shape[1] > collapseSize ):
            print("\tCollapsing Krylov subspace...")
            lastSig = 0
            vSpaceNew = None
            for l in range(k):
                newVect = np.zeros((A.shape[0],1))
                for i in range(vSpace.shape[1]):
                    newVect = newVect + np.dot(eVects[i,l],
                              vSpace[:,i]).reshape((A.shape[0],1))
                # orthonormalize
                newVect = newVect/LIN.norm(newVect)
                if(vSpaceNew is None):
                    vSpaceNew = newVect
                else:
                    vSpaceNew = np.column_stack((vSpaceNew, newVect))
            vSpace = vSpaceNew
            sig = None

        # else, apply preconditioner to residuals (elif)
        else:
            for i in range(k) :
                # apply preconditioner
                sNew = (1.0/(D-eVals[i]))*r[:,i]
                if ( math.sqrt(np.dot(sNew.T, sNew)) > vect_cutoff ):
                    # orthogonalize
                    h = np.dot(vSpace.T, sNew);
                    sNew = sNew - np.dot(vSpace,h);
                    # normalize
                    sNew = (1.0/math.sqrt(np.dot(sNew.T, sNew)))*sNew
                    vSpace = np.column_stack((vSpace, sNew))
                    lastSig = lastSig + 1
        # error check for user adding too many guess vects
        if ( vSpace.shape[1] > A.shape[1] ):
            print("...\nError: Make sure your inputs for Davidson are \
                   reasonable!\n\n");
            exit()
        # save the eigenvalues
        lastVals = eVals
        # increase j and loop again
        j = j+1;

