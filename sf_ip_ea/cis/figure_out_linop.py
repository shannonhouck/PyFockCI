import psi4
import numpy as np
from scipy.sparse.linalg import LinearOperator

# Kronaker delta function.
def kdel(i, j): 
    if(i==j):
        return 1
    else:
        return 0

def mv(v):
    dets = np.load('dets.npy')
    F = np.load('F.npy')
    tei = np.load('tei.npy')
    out = np.zeros((v.shape[0], 1))
    # sigma_0
    #for d_index, det in enumerate(dets):
    #    j = det[0]
    #    b = det[1]
    #    out[0] = out[0] + v[d_index]*F[j,b]
    # excited states
    for d1_index, det1 in enumerate(dets[1:, :]):
        i = det1[0]
        a = det1[1]
        tmp = v[0]*F[i, a]
        for d2_index, det2 in enumerate(dets):
            j = det2[0]
            b = det2[1]
            tmp = tmp + v[d2_index]*(F[a,b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a,j,i,b])
        out[d1_index+1] = tmp
    return np.array(out)

def rmv(v):
    print("idk how this fn works")
    return np.zeros(30)

def mm(v):
    print("matvec called :(")
    return np.zeros(30)


'''
def matmat(v):
    dets = np.load('dets.npy')
    F = np.load('F.npy')
    tei = np.load('tei.npy')
    out = np.zeros((v.shape[0], 1)) 
    for d1_index, det1 in enumerate(dets):
        i = det1[0]
        a = det1[1]
        tmp = v[0]*F[i, a]
        for d2_index, det2 in enumerate(dets):
            j = det2[0]
            b = det2[1]
            tmp = tmp + v[d2_index]*(F[a,b]*kdel(i,j) - F[i,j]*kdel(a,b) + tei[a,j,i,b])
        out[d1_index] = tmp 
    return np.array(out)
'''
