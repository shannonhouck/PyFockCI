import psi4
import numpy as np
from scipy.sparse.linalg import LinearOperator

class LinOpH (LinearOperator):
    

    def __init__(self, shape_in, dets_in, F_in, tei_in):
        super(LinOpH, self).__init__(dtype=np.dtype('float64'), shape=shape_in)
        self.dets = dets_in
        self.F = F_in
        self.tei = tei_in

    def _matvec(self, v):
        dets = self.dets
        F = self.F
        tei = self.tei
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
            for d2_index, det2 in enumerate(dets[1:, :]):
                j = det2[0]
                b = det2[1]
                Ftmp = 0
                if(i==j):
                    Ftmp = Ftmp + F[a,b]
                if(a==b):
                    Ftmp = Ftmp - F[i,j]
                tmp = tmp + v[d2_index+1]*(Ftmp + tei[a,j,i,b])
            out[d1_index+1] = tmp
        return np.array(out)
    
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
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
