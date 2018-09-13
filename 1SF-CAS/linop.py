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
        out = np.zeros((v.shape[0], 1))
        conf_space = self.conf_space
        na_occ = self.na_occ
        nb_occ = self.nb_occ
        na_virt = self.na_virt
        nb_virt = self.nb_virt
        nbf = na_occ + na_virt
        socc = na_occ - nb_occ
        if(conf_space==""):
            v_ref = np.reshape(v, (socc, socc))
        if(conf_space=="p"):
            v_ref = np.reshape(v, (socc, nb_virt))
        # excited states
        out1 = None
        out2 = None
        out3 = None
        # RAS2 -> RAS2
        if(conf_space==""):
            # one-electron part
            v_tmp = v_ref[:socc, :socc]
            Fi_tmp = F[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = F[nbf+nb_occ:nbf+na_occ, nbf+nb_occ:nbf+na_occ]
            F_tmp = np.einsum("ia,aa->ia", v_tmp, Fa_tmp) - np.einsum("ia,ii->ia", v_tmp, Fi_tmp)
            F_tmp = np.reshape(F_tmp, (v.shape[0], 1))
            # two-electron part
            tei_tmp = tei[nbf+nb_occ:nbf+na_occ, nb_occ:na_occ, nb_occ:na_occ, nbf+nb_occ:nbf+na_occ]
            tei_tmp = np.reshape(np.einsum("ia,bija->jb", v_tmp, tei_tmp), (v.shape[0], 1))
            #tei_tmp = np.einsum("ia,bija->jb", v_tmp, tei_tmp).flatten()
            out = F_tmp + tei_tmp
        # RAS2 -> RAS3
        if(conf_space=="p"):
            # one-electron part
            v_tmp = v_ref[:socc, :nb_virt]
            Fi_tmp = F[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = F[nbf+nb_occ:nbf+nbf, nbf+nb_occ:nbf+nbf]
            F_tmp = np.einsum("ia,aa->ia", v_tmp, Fa_tmp) - np.einsum("ia,ii->ia", v_tmp, Fi_tmp)
            F_tmp = F_tmp.flatten()
            # two-electron part
            tei_tmp = tei[nbf+nb_occ:nbf+nbf, nb_occ:na_occ, nb_occ:na_occ, nbf+nb_occ:nbf+nbf]
            tei_tmp = np.einsum("ia,bija->jb", v_tmp, tei_tmp).flatten()
            #tei_tmp = np.einsum("jl,ijkl->ki", v_tmp, tei_tmp).flatten()
            out = F_tmp + tei_tmp
        #if(conf_space=="p"):
        #    v_tmp = v_ref[socc+socc:, socc+na_virt:]
        #    # RAS2 -> RAS3 and RAS2(a) -> RAS2(beta)
        #    for i in range(nb_occ, na_occ):
        #        for a in range(nbf+na_occ, nbf+nbf):
        #            for j in range(nb_occ, i):
        #                for b in range(nbf+na_occ, nbf+nbf):
        #                    out3[] += v_tmp[a,i]*tei[a,i,j,b]
        return out
    
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


