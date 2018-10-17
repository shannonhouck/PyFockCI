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
            F_tmp = np.einsum("ib,ba->ia", v_b1, Fa_tmp) - np.einsum("ja,ji->ia", v_b1, Fi_tmp)
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
            v_b1 = v[:(socc*socc)] # v for block 1
            v_b2 = v[(socc*socc):(socc*nb_virt)] # v for block 2
            v_b3 = v[(socc*nb_virt):] # v for block 3
            # sig(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, socc))
            # sig(2) indexing: (iA:ab)
            v_ref2 = np.reshape(v_b2, (socc, na_virt))
            # sig(3) indexing: (ijBc:aaab)
            v_ref3_1 = np.reshape(v_b3, (socc, socc, na_virt, socc))
            v_ref3_2 = np.reshape(v_b3, (socc, socc, na_virt, socc))

            # enforce antisymmetry
            v_ref3 = 0.5*(v_ref3_1 - v_ref3_2.transpose((1, 0, 2, 3)))

	    #print(v_ref3 + v_ref3.transpose((1,0,2,3)))

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
           
            #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("ib,ba->ia", v_ref1, Fa_tmp) - np.einsum("ja,ji->ia", v_ref1, Fi_tmp)
            #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(iA:ab)*F(aA:bb) - sig(jB:ab)*I(ajBi:baba)
            Fa_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("iA,aA->ia", v_ref2, Fa_tmp)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jB,ajBi->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 

            #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
            #Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            #sig_1 = sig_1 + np.einsum("ijAa,jA->ia", v_ref3, Fa_tmp)
            #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba) + 0.5*v(jkAb:aaab)*I(jkia:aaaa)
            #tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            #sig_1 = sig_1 - 2.0*np.einsum("ijAb,ajbA->ia", v_ref3, tei_tmp)
            #tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            #sig_1 = sig_1 - 0.5*(np.einsum("ijAa,jkAi->ia", v_ref3, tei_tmp_K) - np.einsum("jkAa,jkiA->ia", v_ref3, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
            Fa_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_2 = np.einsum("ib,Ab->iA", v_ref1, Fa_tmp)
            #   sig(iA:ab) += v(jb:ab)*t(jb:ab)*I(Ajbi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("jb,Ajbi->iA", v_ref1, tei_tmp)
            
            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 

            #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fa_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_2 = sig_2 + np.einsum("iB,BA->iA", v_ref2, Fa_tmp) - np.einsum("jA,ji->iA", v_ref2, Fi_tmp)
            #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_2 = sig_2 + np.einsum("jB,AjBi->iA", v_ref2, tei_tmp) # CHECK THIS LATER PLEASE

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_2 = sig_2 - 2.0*np.einsum("ijBc,AjcB->iA", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
            #Fi_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            #sig_3 = np.einsum("jb,iA->ijAb", v_ref1, Fi_tmp) - np.einsum("ib,jA->ijAb", v_ref1, Fi_tmp)
            #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
            #tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            #sig_3 =  -1.0*(np.einsum("kb,Akij->ijAb", v_ref1, tei_tmp) - np.einsum("kb,Akji->ijAb", v_ref1, tei_tmp))
            #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
            #tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            #sig_3 = sig_3 - np.einsum("ic,Abjc->ijAb", v_ref1, tei_tmp) + np.einsum("jc,Abic->ijAb", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_3 = -1.0*np.einsum("iC,AbjC->ijAb", v_ref2, tei_tmp)
            sig_3 = sig_3 + np.einsum("jC,AbiC->ijAb", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
            Fa1_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            Fa2_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("ijAa,ba->ijAb", v_ref3, Fa1_tmp) # no contribution
            sig_3 = sig_3 + np.einsum("ijBb,AB->ijAb", v_ref3, Fa2_tmp) # no contribution
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("ikAb,jk->ijAb", v_ref3, Fi_tmp) + np.einsum("jkAb,ki->ijAb", v_ref3, Fi_tmp)
            #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab) + 0.5*v(klAb:aaab)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            sig_3 = sig_3 + np.einsum("ijBc,AbBc->ijAb", v_ref3, tei_tmp)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 + 0.5*(np.einsum("klAb,klij->ijAb", v_ref3, tei_tmp) - np.einsum("klAb,klji->ijAb", v_ref3, tei_tmp))
            #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab) + v(kiAc:aaab)*I(kbjc:abab)
            sig_3 = sig_3 - np.einsum("kjAc,kbic->ijAb", v_ref3, tei_tmp) + np.einsum("kiAc,kbjc->ijAb", v_ref3, tei_tmp)
            #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa) + v(jkCb:aaab)*I(AkCi:aaaa)
            tei_tmp_J = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            tei_tmp_K = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ, nbf))
            sig_3 = sig_3 - (np.einsum("ikCb,AkCj->ijAb", v_ref3, tei_tmp_J) - np.einsum("ikCb,AkjC->ijAb", v_ref3, tei_tmp_K))
            sig_3 = sig_3 + (np.einsum("jkCb,AkCi->ijAb", v_ref3, tei_tmp_J) - np.einsum("jkCb,AkiC->ijAb", v_ref3, tei_tmp_K))

            ################################################ 
            # sigs complete-- free to reshape!
            ################################################ 
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            sig_3 = np.reshape(sig_3, (v_b3.shape[0], 1))

            # combine and return
            return np.vstack((sig_1, sig_2, sig_3))

    # These two are vestigial-- I'm sure they served some purpose in the parent class,
    # but we only really need matvec for our purposes!
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


