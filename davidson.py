from __future__ import print_function
import numpy as np
from numpy import linalg as LIN

def davidson( A, vInit, cutoff, maxIter ):
    # generate vSpace (search subspace)
    # based on vInit (input vectors)
    vSpace = vInit;
    # size at which to collapse search subspace
    collapseSize = 300;
    # cutoff for adding vector to Krylov search subspace
    delta = 0.000001;
    # number of eigenvalues to solve for
    k = vInit.shape[1];

    # iterations completed
    j = 0;

    sig = None;

    # index of last sigma added
    lastSig = 0;

    while ( j<maxIter ):
        # form k sigma vectors
        for i in range(lastSig, vSpace.shape[1]):
            if(type(sig)==type(None)):
                sig = np.dot(A, vSpace[:,i])
            else:
                sig = np.column_stack((sig, np.dot(A, vSpace[:,i])))
  
        # form subspace matrix
        Av = np.dot(vSpace.T, sig)
        # solve for k lowest eigenvalues
        eVals, eVects = LIN.eig(Av);
        # eIndex tracks where the lowest values are
        # so we can extract the appropriate eigenvectors
        # discard high values
        eIndex = eVals.argsort()[0:k]
        eVals = np.sort(eVals)[0:k]

        r = None;
        # compute residuals
        for i in range(k):
            rNew = np.dot(sig, eVects[:,eIndex[i]]) - eVals[i]*np.dot(vSpace, eVects[:,eIndex[i]]);
            if(type(r)==type(None)):
                r = rNew
            else:
                r = np.column_stack((r, rNew));

        print("Iteration: %4i" % j, end='')
        for i in eVals:
            print("%16.8f" % i, end='')
        for i in range(k):
            print("%16.8f" % LIN.norm(r[:,i]), end='')
        print("\n")

        # check residuals for convergence (break)
        converged = True;
        for i in range(k):
            if( LIN.norm(r[:,i])>cutoff ):
                converged = False

        if ( converged ):
            print("Done! \n")
            return;
        #collapse subspace if necessary (if)
        elif( vSpace.shape[1] > collapseSize ):
            print("uwu")
            lastSig = 0
            vSpaceNew = None
            for l in range(k):
                newVect = np.zeros((rows(A),1))
                for i in range(vSpace.shape[1]):
                    newVect = newVect + np.dot(eVects[i,eIndex(l)], vSpace[:,i])
            # orthonormalize
            newVect = newVect/norm(newVect);
            vSpaceNew = [vSpaceNew, newVect];
            vSpace = vSpaceNew
            sig = None

        # else, apply preconditioner to residuals (elif)
        else:
            # apply preconditioner
            D = np.diag(np.diag(A));
            for i in range(k) :
                sNew = LIN.solve(LIN.inv(D-eVals[i]*np.eye(D.shape[0])), r[:,i])
                print(sNew)
                if ( LIN.norm(sNew) > delta ):
                    # orthogonalize
                    h = np.dot(vSpace.T, sNew);
                    sNew = sNew - np.dot(vSpace,h);
                    # normalize
                    sNew = (1.0/LIN.norm(sNew))*sNew
                    vSpace = np.column_stack((vSpace, sNew));
                    lastSig = lastSig + 1;
                else:
                    print("AAAAAAAAA WHY")
        if ( vSpace.shape[1] > A.shape[1] ):
            printf("...\nError: Make sure your inputs are reasonable!\n\n");
            return;
        # increase j and loop again
        j = j+1;

dim = 2000
ATest = np.random.rand(dim, dim)
ATest = ATest.T + ATest
ATest = ATest + np.diag(np.eye(dim,1)) - 5000*np.eye(dim)
guessindexes = np.diag(ATest).argsort()
vTest = np.zeros((dim,2))
#vTest[guessindexes[0], 0] = 1.0
#vTest[guessindexes[1], 1] = 1.0
vTest[0, 0] = 1.0
vTest[1, 1] = 1.0
print(vTest.shape)
cutTest = 1e-4; # cutoff
maxTest = 300; # max iteration
print("Mine:\n");
davidson(ATest, vTest, cutTest, maxTest);
print("Expected:\n");
print(np.sort(LIN.eigvals(ATest)))

