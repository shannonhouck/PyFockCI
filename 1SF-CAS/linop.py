import psi4
import numpy as np
from scipy.sparse.linalg import LinearOperator

class LinOpH (LinearOperator):
    
    def __init__(self, shape_in, na_occ_in, nb_occ_in, na_virt_in, nb_virt_in, F_in, tei_in, conf_space_in=""):
        super(LinOpH, self).__init__(dtype=np.dtype('float64'), shape=shape_in)
        self.na_occ = na_occ_in
        self.nb_occ = nb_occ_in
        self.na_virt = na_virt_in
        self.nb_virt = nb_virt_in
        self.conf_space = conf_space_in
        self.F = F_in
        self.tei = tei_in

    def _matvec(self, v):
        F = self.F
        tei = self.tei
        out_ref = np.zeros((v.shape[0], 1))
        out = np.zeros((v.shape[0], 1))
        conf_space = self.conf_space
        na_occ = self.na_occ
        nb_occ = self.nb_occ
        na_virt = self.na_virt
        nb_virt = self.nb_virt
        nbf = na_occ + na_virt
        socc = na_occ - nb_occ
        # excited states
        '''
        for i1 in range(nb_occ, na_occ):
            for a1 in range(nbf+nb_occ, nbf+na_occ):
                tmp = 0
                for i2 in range(nb_occ, na_occ):
                    for a2 in range(nbf+nb_occ, nbf+na_occ):
                        Ftmp = 0
                        if(i1==i2):
                            Ftmp = Ftmp + F[a1, a2]
                        if(a1==a2):
                            Ftmp = Ftmp - F[i1, i2]
                        tmp = tmp + v[(i2-nb_occ)*(na_occ - nb_occ)+(a2-(nbf+nb_occ))]*(Ftmp + tei[a1, i2, i1, a2])
                out_ref[(i1-nb_occ)*(na_occ - nb_occ)+(a1-(nbf+nb_occ))] = tmp
        '''
        v_tmp = np.reshape(v, (socc, socc)) # [i, a]
        # one-electron part
        Fi_tmp = F[nb_occ:na_occ, nb_occ:na_occ]
        Fa_tmp = F[nbf+nb_occ:nbf+na_occ, nbf+nb_occ:nbf+na_occ]
        F_tmp = np.einsum("ia,aa->ia", v_tmp, Fa_tmp) - np.einsum("ia,ii->ia", v_tmp, Fi_tmp)
        F_tmp = np.reshape(F_tmp, out.shape)
        # two-electron part
        tei_tmp = tei[nbf+nb_occ:nbf+na_occ, nb_occ:na_occ, nb_occ:na_occ, nbf+nb_occ:nbf+na_occ]
        tei_tmp = np.reshape(np.einsum("ia,bija->jb", v_tmp, tei_tmp), out.shape)
        out = F_tmp + tei_tmp
        ''' DETERMINANT BASED
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
        '''
        return np.array(out)
    
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


