import psi4
import numpy as np
from scipy.sparse.linalg import LinearOperator

class LinOpH (LinearOperator):
    
    def __init__(self, shape_in, na_occ_in, nb_occ_in, na_virt_in, nb_virt_in, Fa_in, Fb_in, tei_in, conf_space_in=""):
        super(LinOpH, self).__init__(dtype=np.dtype('float64'), shape=shape_in)
        # getting the numbers of orbitals
        self.na_occ = na_occ_in # number of alpha occupied
        self.nb_occ = nb_occ_in # number of beta occupied
        self.na_virt = na_virt_in # number of alpha virtual
        self.nb_virt = nb_virt_in # number of beta virtual
        # excitation scheme to use
        self.conf_space = conf_space_in
        # getting integrals
        self.Fa = Fa_in
        self.Fb = Fb_in
        self.tei = tei_in

    def _matvec(self, v):
        # grabbing necessary info from self
        Fa = self.Fa
        Fb = self.Fb
        tei = self.tei
        conf_space = self.conf_space
        na_occ = self.na_occ
        nb_occ = self.nb_occ
        na_virt = self.na_virt
        nb_virt = self.nb_virt
        nbf = na_occ + na_virt
        socc = na_occ - nb_occ
        # do excitation scheme: 1SF-CAS
        if(conf_space==""):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ai:ba)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """

            ################################################ 
            # Put guess vector into block 1
            ################################################ 
            # using reshape because otherwise we can't use v.shape[0] later
            # shouldn't affect too much but if it's an issue, store that value as a variable
            v_b1 = np.reshape(v, (socc,socc)) # v for block 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            # one-electron part (OK!!)
            #   sig(ia:ba) += -v(ia:ba) (eps(a:b)-eps(i:a))
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            F_tmp = np.einsum("ia,ab->ia", v_b1, Fa_tmp) - np.einsum("ia,ij->ia", v_b1, Fi_tmp)
            F_tmp.shape = (v.shape[0], 1)
            # two-electron part (OK!!)
            #   sig(ai:ba) += -v(bj:ba) I(ajbi:baba)
            # using reshape because tei is non-contiguous in memory (look into this while doing speedup)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            tei_tmp = np.reshape(-1.0*np.einsum("jb,ajbi->ia", v_b1, tei_tmp), (v.shape[0], 1))
            return F_tmp + tei_tmp
        # do excitation scheme: 1SF-CAS + h
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
            # Separate guess vector into blocks 1 and 2
            ################################################ 
            v_b1 = v[:(socc*nb_virt)] # v for block 1
            v_b2 = v[(socc*nb_virt):] # v for block 2
            # sig(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, nb_virt))
            # sig(2) indexing: (ijBc:aaab)
            v_ref2 = np.reshape(v_b2, (socc, socc, na_virt, socc))

            ################################################ 
            # PROBABLY OKAY!!
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
           
            #   sig(ia':ba) += v(ia':ab) F(a'b':bb) - v(ia':ab) F(ij:aa)
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = Fb[nb_occ:nbf, nb_occ:nbf]
            sig_1 = np.einsum("ia,ab->ia", v_ref1, Fa_tmp) - np.einsum("ia,ij->ia", v_ref1, Fi_tmp)
            #   sig(a'i:ba) += -v(b'j:ba) I(ja'ib':abab)
            tei_tmp = self.tei.get_subblock((nb_occ, nbf), (nb_occ, na_occ), (nb_occ, nbf), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)
            
            '''
            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            # two-electron part (like 90% probably ok)
            #   sig(a'i:ba) += -2*t(ijBc:aaab) I(a'jcB:baaa)
            tei_tmp = self.tei.get_subblock((nb_occ,nbf), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ,nbf))
            sig_1 = sig_1 - 2.0*np.einsum("ijBc,ajcB->ia", v_ref2, tei_tmp)
            #   sig(ai:ba) += t(kjAa:aaab) I(kjiA:aaaa) (partial block!)
            tei_tmp = self.tei.get_subblock((nb_occ,na_occ), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ,nbf))
            partial_tmp = np.einsum("kjAa,kjiA->ia", v_ref2, tei_tmp)
            tei_tmp = self.tei.get_subblock((nb_occ,na_occ), (nb_occ,na_occ), (na_occ,nbf), (nb_occ,na_occ))
            partial_tmp = partial_tmp - np.einsum("kjAa,kjAi->ia", v_ref2, tei_tmp)
            sig_1 = sig_1 + np.hstack((partial_tmp, np.zeros((partial_tmp.shape[0], na_virt))))

            ################################################ 
            # sig(1) is complete-- free to reshape!
            ################################################ 
            #sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            # two-electron part (wip)
            #   sig(ijAa:aaab) += t(ia':ab)*I(Abja':abaa)
            #tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, nbf))
            #out3 = -1.0*np.einsum("ib,Aajb->ijAa", v_ref1, tei_tmp)
            #   sig(ijAa:aaab) += t(ia:ab)*I(kAij:abaa)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_2 = -0.5*(np.einsum("ka,kAij->ijAa", v_ref1[:, :socc], tei_tmp) - np.einsum("ka,kAji->ijAa", v_ref1[:, :socc], tei_tmp))
            #   sig(ijAa:aaab) += t(ib':ab)*I(Aab'j:abba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (na_occ, nbf), (nb_occ, nbf), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("ib,aAbj->ijAa", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ############################################### 

            # one-electron part (OK??)
            #    sig(ijAa:aaab) += t(ikAb:aaab)*Fi(kj:aa)
            #    sig(ijAa:aaab) += t(ijAb:aaab)*Fa(ab:bb)
            Fi2_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fa2_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("ijAb,ab->ijAb", v_ref2, Fa2_tmp) - np.einsum("ijAb,jk->ijAb", v_ref2, Fi2_tmp)
            # two-electron part (OK)
            #    sig(ijAa:aaab) += t(ijCd:aaab)*I(AaCd:abab)
            tei_tmp = self.tei.get_subblock((na_occ,nbf), (nb_occ,na_occ), (na_occ,nbf), (nb_occ,na_occ))
            sig_2 = sig_2 + np.einsum("ijCd,AaCd->ijAa", v_ref2, tei_tmp)

            #    sig(ijAa:aaab) += 0.5*( t(klAb:aaab)*I(klij:aaaa) )
            tei_tmp = self.tei.get_subblock((nb_occ,na_occ), (nb_occ,na_occ), (nb_occ,na_occ), (nb_occ,na_occ))
            sig_2 = sig_2 + 0.5*(np.einsum("klAa,klij->ijAa", v_ref2, tei_tmp)-np.einsum("klAa,klji->ijAa", v_ref2, tei_tmp))

            #    sig(ijAa:aaab) += t(ikBa:aaab)*I(AkBj:aaaa) - t(jkBa:aaab)*I(AkBi:aaaa)
            # coulomb term
            tei_tmp = self.tei.get_subblock((na_occ,nbf), (nb_occ,na_occ), (na_occ,nbf), (nb_occ,na_occ))
            sig_2 = sig_2 + np.einsum("ikBa,AkBj->ijAa", v_ref2, tei_tmp)
            sig_2 = sig_2 - (np.einsum("jkBa,AkBi->ijAa", v_ref2, tei_tmp))
            # exchange term
            tei_tmp = self.tei.get_subblock((na_occ,nbf), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ,nbf))
            sig_2 = sig_2 - np.einsum("ikBa,AkjB->ijAa", v_ref2, tei_tmp)
            sig_2 = sig_2 + (np.einsum("jkBa,AkiB->ijAa", v_ref2, tei_tmp))
            '''

            ################################################ 
            # sig(2) complete, reshape!
            ################################################ 
            #sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))

            # time to PROBLEM SOLVEEE
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.zeros((v_b2.shape[0], 1))

            # combine and return
            return np.vstack((sig_1, sig_2))

    # These two are vestigial-- I'm sure they served some purpose in the parent class,
    # but we only really need matvec for our purposes!
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


