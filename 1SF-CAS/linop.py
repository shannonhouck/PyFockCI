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
            v_b12 = v[:(socc*np_virt), :] # v for block 1 and block 2
            v_b3 = v[(socc*np_virt):, :] # v for block 3
            v_ref12 = np.reshape(v, (socc, nb_virt))
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
            tei_tmp = np.reshape(np.einsum("jb,jabi->ia", v_tmp, tei_tmp), (v.shape[0], 1))
            out = F_tmp + tei_tmp
        # RAS2 -> RAS3
        if(conf_space=="p"):
            """
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied
                i'     alpha occupied
                a'     beta unuoccupied


                block1 = v(ai:ba)
                block2 = v(Ai:ba)
                block3 = v(Aaij:abaa)

                Because block1 and block2 contain topologically equivalent diagrams, combine and redefine:  

                block1 = v(a'i:ba)
                block2 = v(Aaij:abaa)

                Evaluate the following matrix vector multiply:

                | H(1,1) H(1,2) | * v(1) = sig(1)
                | H(2,1) H(2,2) | * v(2) = sig(2)
           
            """
            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
           
            # one-electron part
            v_tmp = v_ref12
            Fi_tmp = F[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = F[nbf+nb_occ:nbf+nbf, nbf+nb_occ:nbf+nbf]
            F_tmp = np.einsum("ia,aa->ia", v_tmp, Fa_tmp) - np.einsum("ia,ii->ia", v_tmp, Fi_tmp)
            F_tmp = F_tmp.flatten()
            
            # two-electron part
            #   sig(a'i:ba) += -v(b'j:ba) I(ja'ib':abab)

            # get sublock of integrals
            tei_tmp = tei[nbf+nb_occ:nbf+nbf, nb_occ:na_occ, nb_occ:na_occ, nbf+nb_occ:nbf+nbf]
            out1 = np.einsum("jb,jabi->ia", v_tmp, tei_tmp)
            out1.shape = (v.shape[0], 1)
            #out1 = np.einsum("jb,jabi->ia", v_tmp, tei_tmp), (v.shape[0], 1))

            out1 = F_tmp + tei_tmp
            
            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            
            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            
            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 


            # block 3
            v_tmp = v_ref3
            out2 = np.zeros((socc*factorial(socc)/(2*factorial(socc)), 1))
            for i in range(nb_occ, nb_occ+socc):
                for j in range(nb_occ, i):
                    for a in range(nbf+nb_occ, nbf+na_occ): # to beta (socc)
                        # B in alpha virtual, c in beta RAS2
                        for B in range(na_occ, nbf): # to alpha virtual
                            for c in range(nbf+nb_occ, nbf+socc): # to beta in RAS2
                                out2[] = t[]*tei[a,j,B,c] - t[]*tei[a,j,c,b]
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


