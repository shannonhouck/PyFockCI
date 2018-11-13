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
        if(conf_space=="h"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ai:ba)
                block2 = v(Ia:ab)
                block3 = v(Iiab:babb)

                Evaluate the following matrix vector multiply:

                H(1,1) * v(1) + H(1,2) * v(2) + H(1,3) * v(3) = sig(1)
                H(2,1) * v(1) + H(2,2) * v(2) + H(2,3) * v(3) = sig(2)
                H(3,1) * v(1) + H(3,2) * v(2) + H(3,3) * v(3) = sig(3)
            """
            ################################################ 
            # Separate guess vector into blocks 1, 2, and 3
            ################################################ 
            v_b1 = v[:(socc*socc)] # v for block 1
            v_b2 = v[(socc*socc):((socc*socc)+(socc*nb_occ))] # v for block 2
            v_b3 = v[((socc*socc)+(socc*nb_occ)):] # v for block 3
            # v(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, socc))
            # v(2) indexing: (Ia:ab)
            v_ref2 = np.reshape(v_b2, (nb_occ, socc))
            # v(3) unpack to indexing: (Iiab:babb)
            v_ref3 = np.zeros((nb_occ, socc, socc, socc))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            v_ref3[I, i, a, b] = v_b3[index]
                            v_ref3[I, i, b, a] = -1.0*v_b3[index]
                            index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("ib,ab->ia", v_ref1, Fb_tmp) - np.einsum("ja,ji->ia", v_ref1, Fa_tmp)
            #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Ia,Ii->ia", v_ref2, Fa_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("Ib,Iaib->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_1 = sig_1 + np.einsum("Iiba,bI->ia", v_ref3, Fb_tmp)
            #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("Ijba,Ijbi->ia", v_ref3, tei_tmp)
            #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 + 0.5*(np.einsum("Iicb,Iacb->ia", v_ref3, tei_tmp) - np.einsum("Iicb,Iabc->ia", v_ref3, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
            Fa_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("ia,iI->Ia", v_ref1, Fa_tmp)
            #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("ib,iaIb->Ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += sig(Ia:ab)*F(ab:bb) - sig(Ja:ab)*F(IJ:aa)
            Fa_tmp = Fa[0:nb_occ, 0:nb_occ]
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("Ib,ab->Ia", v_ref2, Fb_tmp) - np.einsum("Ja,IJ->Ia", v_ref2, Fa_tmp)
            #   sig(Ia:ab) += v(Jb:ab)*I(JaIb:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("Jb,JaIb->Ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -v(Jiab:babb)*I(JibI:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_2 = sig_2 + np.einsum("Jiab,JibI->Ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_3 = (np.einsum("ib,Ia->Iiab", v_ref1, Fb_tmp) - np.einsum("ia,Ib->Iiab", v_ref1, Fb_tmp))
            #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_3 = sig_3 + np.einsum("ja,jbiI->Iiab", v_ref1, tei_tmp) - np.einsum("jb,jaiI->Iiab", v_ref1, tei_tmp)
            #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_3 = sig_3 + np.einsum("ic,abIc->Iiab", v_ref1, tei_tmp_J) - np.einsum("ic,abcI->Iiab", v_ref1, tei_tmp_K)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_3 = sig_3 + np.einsum("Ja,JbiI->Iiab", v_ref2, tei_tmp)
            sig_3 = sig_3 - np.einsum("Jb,JaiI->Iiab", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(Iiab:babb) += t(Iiac:babb)*F(bc:bb) - t(Iibc:babb)*F(ac:bb)
            F_ac_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Iiac,bc->Iiab", v_ref3, F_ac_tmp) - np.einsum("Iibc,ac->Iiab", v_ref3, F_ac_tmp)
            #   sig(Iiab:babb) += -1.0*t(Ijab:babb)*F(ij:aa)
            F_ij_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Ijab,ij->Iiab", v_ref3, F_ij_tmp)
            #   sig(Iiab:babb) += -1.0*t(Jiab:babb)*F(IJ:bb)
            F_IJ_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_3 = sig_3 - np.einsum("Jiab,IJ->Iiab", v_ref3, F_IJ_tmp)
            #   sig(Iiab:babb) += 0.5*v(Iicd:babb)*I(abcd:bbbb)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 + 0.5*(np.einsum("Iicd,abcd->Iiab", v_ref3, tei_tmp) - np.einsum("Iicd,abdc->Iiab", v_ref3, tei_tmp))
            #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 - np.einsum("Ijcb,ajci->Iiab", v_ref3, tei_tmp) + np.einsum("Ijca,bjci->Iiab", v_ref3, tei_tmp)
            #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_3 = sig_3 - (np.einsum("Jiac,JbIc->Iiab", v_ref3, tei_tmp_J) - np.einsum("Jiac,JbcI->Iiab", v_ref3, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_3 = sig_3 + (np.einsum("Jibc,JaIc->Iiab", v_ref3, tei_tmp_J) - np.einsum("Jibc,JacI->Iiab", v_ref3, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_3 = sig_3 + np.einsum("Jjab,JjIi->Iiab", v_ref3, tei_tmp)

            ################################################ 
            # sigs complete-- free to reshape!
            ################################################ 
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            # pack sig(3) vector for returning
            sig_3_out = np.zeros((v_b3.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            sig_3_out[index] = sig_3[I, i, a, b]
                            index = index + 1

            # combine and return
            return np.vstack((sig_1, sig_2, sig_3_out))

        # do excitation scheme: 1SF-CAS + p
        # rearrange later to take advantage of Aaij indexing
        if(conf_space=="p"):
            """
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ai:ba)
                block2 = v(Ai:ba)
                block3 = v(Aaij:abaa)

                Evaluate the following matrix vector multiply:

                H(1,1) * v(1) + H(1,2) * v(2) + H(1,3) * v(3) = sig(1)
                H(2,1) * v(1) + H(2,2) * v(2) + H(2,3) * v(3) = sig(2)
                H(3,1) * v(1) + H(3,2) * v(2) + H(3,3) * v(3) = sig(3)
           
            """
            ################################################ 
            # Separate guess vector into blocks 1, 2, and 3
            ################################################ 
            v_b1 = v[:(socc*socc)] # v for block 1
            v_b2 = v[(socc*socc):(socc*nb_virt)] # v for block 2
            v_b3 = v[(socc*nb_virt):] # v for block 3
            # v(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, socc))
            # v(2) indexing: (iA:ab)
            v_ref2 = np.reshape(v_b2, (socc, na_virt))
            # v(3) unpack to indexing: (ijAb:aaab)
            v_ref3 = np.zeros((socc, socc, na_virt, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for k in range(na_virt):
                        for l in range(socc):
                            v_ref3[i, j, k, l] = v_b3[index]
                            v_ref3[j, i, k, l] = -1.0*v_b3[index]
                            index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
           
            #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("ib,ab->ia", v_ref1, Fb_tmp) - np.einsum("ja,ji->ia", v_ref1, Fa_tmp)
            #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
            Fb_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("iA,aA->ia", v_ref2, Fb_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jB,ajBi->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 

            #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("jiAa,jA->ia", v_ref3, Fa_tmp)
            #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_1 = sig_1 - np.einsum("ijAb,ajbA->ia", v_ref3, tei_tmp)
            #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_1 = sig_1 - 0.5*(np.einsum("jkAa,jkAi->ia", v_ref3, tei_tmp) - np.einsum("jkAa,jkiA->ia", v_ref3, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
            Fb_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_2 = np.einsum("ib,Ab->iA", v_ref1, Fb_tmp)
            #   sig(iA:ab) += v(jb:ab)*t(jb:ab)*I(Ajbi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("jb,Ajbi->iA", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 

            #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_2 = sig_2 + np.einsum("iB,AB->iA", v_ref2, Fb_tmp) - np.einsum("jA,ji->iA", v_ref2, Fa_tmp)
            #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("jB,AjBi->iA", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_2 = sig_2 - np.einsum("ijBc,AjcB->iA", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_3 = (np.einsum("jb,iA->ijAb", v_ref1, Fa_tmp) - np.einsum("ib,jA->ijAb", v_ref1, Fa_tmp))
            #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 - np.einsum("ic,Abjc->ijAb", v_ref1, tei_tmp) + np.einsum("jc,Abic->ijAb", v_ref1, tei_tmp)
            #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 - (np.einsum("kb,Akij->ijAb", v_ref1, tei_tmp) - np.einsum("kb,Akji->ijAb", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_3 = sig_3 - np.einsum("iC,AbjC->ijAb", v_ref2, tei_tmp)
            sig_3 = sig_3 + np.einsum("jC,AbiC->ijAb", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
            F_bc_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            F_AB_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("ijAc,bc->ijAb", v_ref3, F_bc_tmp) # no contribution
            sig_3 = sig_3 + np.einsum("ijBb,AB->ijAb", v_ref3, F_AB_tmp) # no contribution
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("ikAb,kj->ijAb", v_ref3, Fi_tmp) + np.einsum("jkAb,ki->ijAb", v_ref3, Fi_tmp)
            #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            sig_3 = sig_3 + np.einsum("ijBc,AbBc->ijAb", v_ref3, tei_tmp)
            #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 + 0.5*(np.einsum("klAb,klij->ijAb", v_ref3, tei_tmp) - np.einsum("klAb,klji->ijAb", v_ref3, tei_tmp))
            #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
            sig_3 = sig_3 - np.einsum("kjAc,kbic->ijAb", v_ref3, tei_tmp) 
            #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
            sig_3 = sig_3 + np.einsum("kiAc,kbjc->ijAb", v_ref3, tei_tmp)
            #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
            tei_tmp_J = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            tei_tmp_K = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ, nbf))
            sig_3 = sig_3 - (np.einsum("ikCb,AkCj->ijAb", v_ref3, tei_tmp_J) - np.einsum("ikCb,AkjC->ijAb", v_ref3, tei_tmp_K))
            #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
            sig_3 = sig_3 + (np.einsum("jkCb,AkCi->ijAb", v_ref3, tei_tmp_J) - np.einsum("jkCb,AkiC->ijAb", v_ref3, tei_tmp_K))

            ################################################ 
            # sigs complete-- free to reshape!
            ################################################ 
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            # pack sig(3) vector for returning
            sig_3_out = np.zeros((v_b3.shape[0], 1)) # add 0.5
            index = 0 
            for i in range(socc):
                for j in range(i):
                    for k in range(na_virt):
                        for l in range(socc):
                            sig_3_out[index] = sig_3[i, j, k, l]  
                            index = index + 1 

            # combine and return
            return np.vstack((sig_1, sig_2, sig_3_out))

        # do excitation scheme: RAS(h,p)-1SF
        if(conf_space=="h,p"):
            """
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ia:ab)
                block2 = v(Ia:ab)
                block3 = v(iA:ab)
                block4 = v(Iiab:babb)
                block5 = v(ijAb:abaa)

                Evaluate the following matrix vector multiply:

                H(1,1) * v(1) + H(1,2) * v(2) + H(1,3) * v(3) + H(1,4) * v(4) + H(1,5) * v(5) = sig(1)
                H(2,1) * v(1) + H(2,2) * v(2) + H(2,3) * v(3) + H(2,4) * v(4) + H(2,5) * v(5) = sig(2)
                H(3,1) * v(1) + H(3,2) * v(2) + H(3,3) * v(3) + H(3,4) * v(4) + H(3,5) * v(5) = sig(3)
                H(4,1) * v(1) + H(4,2) * v(2) + H(4,3) * v(3) + H(4,4) * v(4) + H(4,5) * v(5) = sig(4)
                H(5,1) * v(1) + H(5,2) * v(2) + H(5,3) * v(3) + H(5,4) * v(4) + H(5,5) * v(5) = sig(5)
            """
            ################################################ 
            # Separate guess vector into blocks 1, 2, and 3
            ################################################ 
            n_b1_dets = socc*socc
            n_b2_dets = nb_occ*socc
            n_b3_dets = socc*na_virt
            n_b4_dets = nb_occ*socc*(socc*(socc-1)/2)
            n_b5_dets = na_virt*socc*(socc*(socc-1)/2)
            v_b1 = v[0:n_b1_dets] # v for block 1
            v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets] # v for block 2
            v_b3 = v[n_b1_dets+n_b2_dets:n_b1_dets+n_b2_dets+n_b3_dets] # v for block 3
            v_b4 = v[n_b1_dets+n_b2_dets+n_b3_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets] # v for block 4
            v_b5 = v[n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets+n_b5_dets] # v for block 5
            # v(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, socc))
            # v(2) indexing: (Ia:ab)
            v_ref2 = np.reshape(v_b2, (nb_occ, socc))
            # v(3) indexing: (iA:ab)
            v_ref3 = np.reshape(v_b3, (socc, na_virt))
            # v(3) unpack to indexing: (Iiab:babb)
            v_ref4 = np.zeros((nb_occ, socc, socc, socc))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            v_ref4[I, i, a, b] = v_b4[index]
                            v_ref4[I, i, b, a] = -1.0*v_b4[index]
                            index = index + 1
            # v(5) unpack to indexing: (ijAb:aaab)
            v_ref5 = np.zeros((socc, socc, na_virt, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for k in range(na_virt):
                        for l in range(socc):
                            v_ref5[i, j, k, l] = v_b5[index]
                            v_ref5[j, i, k, l] = -1.0*v_b5[index]
                            index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("ib,ab->ia", v_ref1, Fb_tmp) - np.einsum("ja,ji->ia", v_ref1, Fa_tmp)
            #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Ia,Ii->ia", v_ref2, Fa_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("Ib,Iaib->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            
            #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
            Fb_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("iA,aA->ia", v_ref3, Fb_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("jB,ajBi->ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,4) v(4) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_1 = sig_1 + np.einsum("Iiba,bI->ia", v_ref4, Fb_tmp)
            #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - np.einsum("Ijba,Ijbi->ia", v_ref4, tei_tmp)
            #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 + 0.5*(np.einsum("Iicb,Iacb->ia", v_ref4, tei_tmp) - np.einsum("Iicb,Iabc->ia", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,5) v(5) = sig(1)
            ################################################ 

            #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("jiAa,jA->ia", v_ref5, Fa_tmp)
            #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_1 = sig_1 - np.einsum("ijAb,ajbA->ia", v_ref5, tei_tmp)
            #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_1 = sig_1 - 0.5*(np.einsum("jkAa,jkAi->ia", v_ref5, tei_tmp) - np.einsum("jkAa,jkiA->ia", v_ref5, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
            Fa_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("ia,iI->Ia", v_ref1, Fa_tmp)
            #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("ib,iaIb->Ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += sig(Ia:ab)*F(ab:bb) - sig(Ja:ab)*F(IJ:aa)
            Fa_tmp = Fa[0:nb_occ, 0:nb_occ]
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("Ib,ab->Ia", v_ref2, Fb_tmp) - np.einsum("Ja,IJ->Ia", v_ref2, Fa_tmp)
            #   sig(Ia:ab) += v(Jb:ab)*I(JaIb:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_2 = sig_2 - np.einsum("Jb,JaIb->Ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(iA:ab)*I(aiAI:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (0, nb_occ))
            sig_2 = sig_2 - np.einsum("iA,aiAI->Ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,4) v(4) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += - v(Jiab:babb)*I(JibI:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_2 = sig_2 + np.einsum("Jiab,JibI->Ia", v_ref4, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,5) v(5) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += - v(ijAa:aaab)*I(ijAI:aaaa)
            tei_tmp_J = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (0, nb_occ))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (na_occ, nbf))
            sig_2 = sig_2 - 0.5*(np.einsum("ijAa,ijAI->Ia", v_ref5, tei_tmp_J) - np.einsum("ijAa,ijIA->Ia", v_ref5, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
            Fb_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_3 = np.einsum("ib,Ab->iA", v_ref1, Fb_tmp)
            #   sig(iA:ab) += v(jb:ab)*I(Ajbi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 - np.einsum("jb,Ajbi->iA", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(iA:ab) += -v(Ia:ab)*I(AIai:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 - np.einsum("Ia,AIai->iA", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("iB,AB->iA", v_ref3, Fb_tmp) - np.einsum("jA,ji->iA", v_ref3, Fa_tmp)
            #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (na_occ, nbf), (nb_occ, na_occ))
            sig_3 = sig_3 - np.einsum("jB,AjBi->iA", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,4) v(4) = sig(3)
            ################################################ 

            #   sig(iA:ab) += v(Iibc:babb)*I(IAbc:bbbb)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_3 = sig_3 + 0.5*(np.einsum("Iibc,IAbc->iA", v_ref4, tei_tmp) - np.einsum("Iibc,IAcb->iA", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(3,5) v(5) = sig(3)
            ################################################ 

            #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_3 = sig_3 - np.einsum("ijBc,AjcB->iA", v_ref5, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,1) v(1) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_4 = (np.einsum("ib,Ia->Iiab", v_ref1, Fb_tmp) - np.einsum("ia,Ib->Iiab", v_ref1, Fb_tmp))
            #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_4 = sig_4 + np.einsum("ja,jbiI->Iiab", v_ref1, tei_tmp) - np.einsum("jb,jaiI->Iiab", v_ref1, tei_tmp)
            #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_4 = sig_4 + np.einsum("ic,abIc->Iiab", v_ref1, tei_tmp_J) - np.einsum("ic,abcI->Iiab", v_ref1, tei_tmp_K)

            ################################################ 
            # Do the following term:
            #       H(4,2) v(2) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_4 = sig_4 + np.einsum("Ja,JbiI->Iiab", v_ref2, tei_tmp)
            sig_4 = sig_4 - np.einsum("Jb,JaiI->Iiab", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,3) v(3) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(iA:ab)*I(abIA:bbbb)
            tei_tmp_J = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ), (na_occ, nbf))
            tei_tmp_K = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (0, nb_occ))
            sig_4 = sig_4 + (np.einsum("iA,abIA->Iiab", v_ref3, tei_tmp_J) - np.einsum("iA,abAI->Iiab", v_ref3, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(4,4) v(4) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += t(Iiac:babb)*F(bc:bb) - t(Iibc:babb)*F(ac:bb)
            F_ac_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_4 = sig_4 + np.einsum("Iiac,bc->Iiab", v_ref4, F_ac_tmp) - np.einsum("Iibc,ac->Iiab", v_ref4, F_ac_tmp)
            #   sig(Iiab:babb) += -1.0*t(Ijab:babb)*F(ij:aa)
            F_ij_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_4 = sig_4 - np.einsum("Ijab,ij->Iiab", v_ref4, F_ij_tmp)
            #   sig(Iiab:babb) += -1.0*t(Jiab:babb)*F(IJ:bb)
            F_IJ_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_4 = sig_4 - np.einsum("Jiab,IJ->Iiab", v_ref4, F_IJ_tmp)
            #   sig(Iiab:babb) += 0.5*v(Iicd:babb)*I(abcd:bbbb)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_4 = sig_4 + 0.5*(np.einsum("Iicd,abcd->Iiab", v_ref4, tei_tmp) - np.einsum("Iicd,abdc->Iiab", v_ref4, tei_tmp))
            #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_4 = sig_4 - np.einsum("Ijcb,ajci->Iiab", v_ref4, tei_tmp) + np.einsum("Ijca,bjci->Iiab", v_ref4, tei_tmp)
            #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_4 = sig_4 - (np.einsum("Jiac,JbIc->Iiab", v_ref4, tei_tmp_J) - np.einsum("Jiac,JbcI->Iiab", v_ref4, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
            tei_tmp_J = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            tei_tmp_K = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ), (0, nb_occ))
            sig_4 = sig_4 + (np.einsum("Jibc,JaIc->Iiab", v_ref4, tei_tmp_J) - np.einsum("Jibc,JacI->Iiab", v_ref4, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
            tei_tmp = self.tei.get_subblock((0, nb_occ), (nb_occ, na_occ), (0, nb_occ), (nb_occ, na_occ))
            sig_4 = sig_4 + np.einsum("Jjab,JjIi->Iiab", v_ref4, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,5) v(5) = sig(4) (NC)
            ################################################ 

            #   sig(Iiab:babb) += v(jiAa:aaab)*I(jbAI:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf), (0, nb_occ))
            sig_4 = sig_4 - (np.einsum("jiAa,jbAI->Iiab", v_ref5, tei_tmp) - np.einsum("jiAb,jaAI->Iiab", v_ref5, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,1) v(1) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_5 = (np.einsum("jb,iA->ijAb", v_ref1, Fa_tmp) - np.einsum("ib,jA->ijAb", v_ref1, Fa_tmp))
            #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_5 = sig_5 - np.einsum("ic,Abjc->ijAb", v_ref1, tei_tmp) + np.einsum("jc,Abic->ijAb", v_ref1, tei_tmp)
            #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_5 = sig_5 - (np.einsum("kb,Akij->ijAb", v_ref1, tei_tmp) - np.einsum("kb,Akji->ijAb", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,2) v(2) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += -v(Ib:ab)*I(AIij:aaaa)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_5 = sig_5 - (np.einsum("Ib,AIij->ijAb", v_ref2, tei_tmp) - np.einsum("Ib,AIji->ijAb", v_ref2, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,3) v(3) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ, na_occ), (nb_occ, na_occ), (na_occ, nbf))
            sig_5 = sig_5 - np.einsum("iC,AbjC->ijAb", v_ref3, tei_tmp)
            sig_5 = sig_5 + np.einsum("jC,AbiC->ijAb", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(5,4) v(4) = sig(5)
            ################################################

            #   sig(ijAb:aaab) += - v(Iicb:babb)*I(Abjc:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (0, nb_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_5 = sig_5 - (np.einsum("Iicb,AIjc->ijAb", v_ref4, tei_tmp) - np.einsum("Ijcb,AIic->ijAb", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,5) v(5) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
            F_bc_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            F_AB_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_5 = sig_5 + np.einsum("ijAc,bc->ijAb", v_ref5, F_bc_tmp) # no contribution
            sig_5 = sig_5 + np.einsum("ijBb,AB->ijAb", v_ref5, F_AB_tmp) # no contribution
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_5 = sig_5 - np.einsum("ikAb,kj->ijAb", v_ref5, Fi_tmp) + np.einsum("jkAb,ki->ijAb", v_ref5, Fi_tmp)
            #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
            tei_tmp = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            sig_5 = sig_5 + np.einsum("ijBc,AbBc->ijAb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_5 = sig_5 + 0.5*(np.einsum("klAb,klij->ijAb", v_ref5, tei_tmp) - np.einsum("klAb,klji->ijAb", v_ref5, tei_tmp))
            #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
            sig_5 = sig_5 - np.einsum("kjAc,kbic->ijAb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
            sig_5 = sig_5 + np.einsum("kiAc,kbjc->ijAb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
            tei_tmp_J = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (na_occ, nbf), (nb_occ,na_occ))
            tei_tmp_K = self.tei.get_subblock((na_occ, nbf), (nb_occ,na_occ), (nb_occ,na_occ), (na_occ, nbf))
            sig_5 = sig_5 - (np.einsum("ikCb,AkCj->ijAb", v_ref5, tei_tmp_J) - np.einsum("ikCb,AkjC->ijAb", v_ref5, tei_tmp_K))
            #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
            sig_5 = sig_5 + (np.einsum("jkCb,AkCi->ijAb", v_ref5, tei_tmp_J) - np.einsum("jkCb,AkiC->ijAb", v_ref5, tei_tmp_K))

            ################################################ 
            # sigs complete-- free to reshape!
            ################################################ 
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            sig_3 = np.reshape(sig_3, (v_b3.shape[0], 1))
            # pack sig(4) vector for returning
            sig_4_out = np.zeros((v_b4.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            sig_4_out[index] = sig_4[I, i, a, b]
                            index = index + 1
            # pack sig(5) vector for returning
            sig_5_out = np.zeros((v_b5.shape[0], 1))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for k in range(na_virt):
                        for l in range(socc):
                            sig_5_out[index] = sig_5[i, j, k, l]
                            index = index + 1


            # combine and return
            return np.vstack((sig_1, sig_2, sig_3, sig_4_out, sig_5_out))

        if(conf_space=="EA"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(a:b)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("b,ab->a", v, F_tmp)
            return sig_1

        if(conf_space=="IP"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(i:a)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*np.einsum("j,ji->i", v, F_tmp)
            return sig_1

        # do excitation scheme: 1SF-CAS-EA
        if(conf_space=="CAS_EA"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ija:aab)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            # v(1) unpack to indexing: (ija:aab)
            v_ref1 = np.zeros((socc,socc,socc))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref1[i, a, b] = v[index]
                        v_ref1[i, b, a] = -1.0*v[index]
                        index = index + 1 
            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(ija:aab) += -P(ab)*t(iab:abb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = (np.einsum("icb,ac->iab", v_ref1, Fb_tmp) - np.einsum("ica,bc->iab", v_ref1, Fb_tmp))
            #   sig(ija:aab) += t(jab:abb)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("jab,ji->iab", v_ref1, Fa_tmp)
            #   sig(ija:aab) += P(ab)*t(jac:abb)*I(jbic:abab)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - (np.einsum("jac,jbic->iab", v_ref1, tei_tmp) - np.einsum("jbc,jaic->iab", v_ref1, tei_tmp))
            #   sig(ija:aab) += -0.5*t(icd:abb)*I(abcd:abab)
            sig_1 = sig_1 + 0.5*(np.einsum("icd,abcd->iab", v_ref1, tei_tmp) - np.einsum("icd,abdc->iab", v_ref1, tei_tmp))

            sig_1_out = np.zeros((v.shape[0], 1))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        sig_1_out[index] = sig_1[i, a, b]
                        index = index + 1

            return sig_1_out

        # do excitation scheme: 1SF-CAS-IP
        if(conf_space=="CAS_IP"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ija:aab)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            # v(1) unpack to indexing: (ija:aab)
            v_ref1 = np.zeros((socc,socc,socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        v_ref1[i, j, a] = v[index]
                        v_ref1[j, i, a] = -1.0*v[index]
                        index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(ija:aab) += -P(ij)*v(kja:aab)*F(ki:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*(np.einsum("kja,ki->ija", v_ref1, Fa_tmp) - np.einsum("kia,kj->ija", v_ref1, Fa_tmp))
            #   sig(ija:aab) += v(ijb:aab)*F(ab:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("ijb,ab->ija", v_ref1, Fb_tmp)
            #   sig(ija:aab) += -P(ij)*v(ikb:aab)*I(akbj:baba)
            tei_tmp = self.tei.get_subblock((nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ), (nb_occ, na_occ))
            sig_1 = sig_1 - (np.einsum("ikb,akbj->ija", v_ref1, tei_tmp) - np.einsum("jkb,akbi->ija", v_ref1, tei_tmp))
            #   sig(ija:aab) += 0.5*v(kla:aab)*I(klij:aaaa)
            sig_1 = sig_1 + 0.5*(np.einsum("kla,klij->ija", v_ref1, tei_tmp) - np.einsum("kla,klji->ija", v_ref1, tei_tmp))

            sig_1_out = np.zeros((v.shape[0], 1))
            index = 0 
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        sig_1_out[index] = sig_1[i, j, a]
                        index = index + 1 

            return sig_1_out

    # These two are vestigial-- I'm sure they served some purpose in the parent class,
    # but we only really need matvec for our purposes!
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


