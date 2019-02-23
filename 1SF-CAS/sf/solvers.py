from __future__ import print_function
import numpy as np
from scipy import linalg as LIN

# Solves for the eigenvalues and eigenvectors of a Hamiltonian.
# Parameters:
#    A               Hamiltonian (LinOp object)
#    vInit           Initial guess vector (numpy)
#    e_conv          Cutoff for energy convergence (default: 1e-6)
#    r_conv          Cutoff for residual squared convergence (default: 1e-4)
#    vect_cutoff     Cutoff for adding vectors to Krylov space (default: 1e-8)
#    maxIter         Maximum number of iterations
# Returns:
#    s2              The S**2 expectation value for the state
def davidson( A, vInit, e_conv=1e-6, r_conv=1e-4, vect_cutoff=1e-8, maxIter=100, collapseSize=50 ):
    # initialize vSpace (search subspace)
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

    # index of last sigma added
    lastSig = 0;

    print("Starting Davidson...")
    while ( j<maxIter ):
        # form k sigma vectors
        for i in range(lastSig, vSpace.shape[1]):
            if(type(sig)==type(None)):
                print(vSpace[:,i].shape)
                sig = A.matvec(vSpace[:,i])
                sig = sig.reshape((vSpace.shape[0],1))
            else:
                sig = np.column_stack((sig, A.matvec(vSpace[:,i])))
  
        # form subspace matrix
        Av = np.dot(vSpace.T, sig)
        # solve for k lowest eigenvalues/vectors
        eVals, eVects = LIN.eigh(Av)
        eVals = eVals[:k]
        eVects = eVects[:, :k]

        r = None;
        # compute residuals
        for i in range(k):
            rNew = np.dot(sig, eVects[:,i]) - eVals[i]*np.dot(vSpace, eVects[:,i]);
            if(type(r)==type(None)):
                r = rNew
            else:
                r = np.column_stack((r, rNew));
        print(r.shape)

        # print info about this iteration
        print("\nIteration: %4i" % j)
        for i, val in enumerate(eVals):
            print("\tROOT %2i: %16.8f\t%E" % (i, val, LIN.norm(r[:,i])))

        # check residuals for convergence
        converged = True;
        for i in range(k):
            # check residual
            if( LIN.norm(r[:,i])>r_conv ):
                converged = False
            # check energy
            if( (eVals[i]-lastVals[i] ) > e_conv):
                converged = False

        # if converged, return appropriate values
        if ( converged ):
            print("\nDavidson Converged! \n")
            print("NOTE: still working on eigenvectors.")
            return (eVals, vSpace[:,k:])
        # collapse subspace if necessary (if)
        elif( vSpace.shape[1] > collapseSize ):
            print("\tCollapsing Krylov subspace...")
            lastSig = 0
            vSpaceNew = None
            for l in range(k):
                newVect = np.zeros((A.shape[0],1))
                for i in range(vSpace.shape[1]):
                    newVect = newVect + np.dot(eVects[i,l], vSpace[:,i]).reshape((A.shape[0],1))
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
                sNew = LIN.solve(LIN.inv(D-eVals[i]*np.eye(D.shape[0])), r[:,i])
                if ( LIN.norm(sNew) > vect_cutoff ):
                    # orthogonalize
                    h = np.dot(vSpace.T, sNew);
                    sNew = sNew - np.dot(vSpace,h);
                    # normalize
                    sNew = (1.0/LIN.norm(sNew))*sNew
                    vSpace = np.column_stack((vSpace, sNew))
                    #lastSig = lastSig + 1
                else:
                    pass
                lastSig = lastSig + 1
        # error check for user adding too many guess vects
        if ( vSpace.shape[1] > A.shape[1] ):
            print("...\nError: Make sure your inputs for Davidson are reasonable!\n\n");
            exit()
        # save the eigenvalues
        lastVals = eVals
        # increase j and loop again
        j = j+1;

