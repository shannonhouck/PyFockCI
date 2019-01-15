import psi4
import numpy as np
from scipy.sparse.linalg import LinearOperator

class LinOpH (LinearOperator):
    
    def __init__(self, shape_in, na_occ_in, nb_occ_in, na_virt_in, nb_virt_in, Fa_in, Fb_in, tei_in, n_SF_in, delta_ec_in, conf_space_in=""):
        super(LinOpH, self).__init__(dtype=np.dtype('float64'), shape=shape_in)
        # getting the numbers of orbitals
        self.na_occ = na_occ_in # number of alpha occupied
        self.nb_occ = nb_occ_in # number of beta occupied
        self.na_virt = na_virt_in # number of alpha virtual
        self.nb_virt = nb_virt_in # number of beta virtual
        # setting useful parameters
        self.n_SF = n_SF_in # number of spin-flips
        self.delta_ec = delta_ec_in # change in electron count
        self.conf_space = conf_space_in # excitation rank
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
        n_SF = self.n_SF
        delta_ec = self.delta_ec
        nbf = na_occ + na_virt
        socc = na_occ - nb_occ
        # do excitation scheme: 1SF-CAS
        if(n_SF==1 and delta_ec==0 and conf_space==""):
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            tei_tmp = np.reshape(-1.0*np.einsum("jb,ajbi->ia", v_b1, tei_tmp), (v.shape[0], 1))
            return F_tmp + tei_tmp

        # do excitation scheme: 2SF-CAS
        if(n_SF==2 and delta_ec==0 and conf_space==""):
            """  
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ijab:aabb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            # v(1) unpack to indexing: (ijab:aabb)
            v_ref1 = np.zeros((socc, socc, socc, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        for b in range(a):
                            v_ref1[i, j, a, b] = v[index]
                            v_ref1[i, j, b, a] = -1.0*v[index]
                            v_ref1[j, i, a, b] = -1.0*v[index]
                            v_ref1[j, i, b, a] = v[index]
                            index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(ijab:aabb) += -v(kjab:aabb) F(ik:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*np.einsum("kjab,ki->ijab", v_ref1, Fa_tmp)
            sig_1 = sig_1 + np.einsum("kiab,kj->ijab", v_ref1, Fa_tmp) #P(ij)
            #   sig(ijab:aabb) += v(ijcb:aabb) F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("ijcb,ac->ijab", v_ref1, Fb_tmp)
            sig_1 = sig_1 - np.einsum("ijca,bc->ijab", v_ref1, Fb_tmp) #P(ab)
            #   sig(ijab:aabb) += 0.5*v(ijcd:aabb) I(abcd:bbbb)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 + 0.5*(np.einsum("ijcd,abcd->ijab", v_ref1, tei_tmp) - np.einsum("ijcd,abdc->ijab", v_ref1, tei_tmp))
            #   sig(ijab:aabb) += 0.5*v(klab:aabb) I(klij:aaaa)
            sig_1 = sig_1 + 0.5*(np.einsum("klab,klij->ijab", v_ref1, tei_tmp) - np.einsum("klab,klji->ijab", v_ref1, tei_tmp))
            #   sig(ijab:aabb) += - P(ij)P(ab) v(ikcb:aabb) I(akcj:baba)
            sig_1 = sig_1 - (np.einsum("ikcb,akcj->ijab", v_ref1, tei_tmp))
            sig_1 = sig_1 + (np.einsum("jkcb,akci->ijab", v_ref1, tei_tmp)) #P(ij)
            sig_1 = sig_1 + (np.einsum("ikca,bkcj->ijab", v_ref1, tei_tmp)) #P(ab)
            sig_1 = sig_1 - (np.einsum("jkca,bkci->ijab", v_ref1, tei_tmp)) #P(ij)P(ab)

            sig_1_out = np.zeros((v.shape[0], 1))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        for b in range(a):
                            sig_1_out[index] = sig_1[i, j, a, b]
                            index = index + 1

            return sig_1_out

        # do excitation scheme: 1SF-CAS + h
        if(n_SF==1 and delta_ec==0 and conf_space=="h"):
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Ia,Ii->ia", v_ref2, Fa_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ib,Iaib->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_1 = sig_1 + np.einsum("Iiba,bI->ia", v_ref3, Fb_tmp)
            #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ijba,Ijbi->ia", v_ref3, tei_tmp)
            #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + 0.5*(np.einsum("Iicb,Iacb->ia", v_ref3, tei_tmp) - np.einsum("Iicb,Iabc->ia", v_ref3, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
            Fa_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("ia,iI->Ia", v_ref1, Fa_tmp)
            #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
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
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_2 = sig_2 - np.einsum("Jb,JaIb->Ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -v(Jiab:babb)*I(JibI:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 + np.einsum("Jiab,JibI->Ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_3 = (np.einsum("ib,Ia->Iiab", v_ref1, Fb_tmp) - np.einsum("ia,Ib->Iiab", v_ref1, Fb_tmp))
            #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
            sig_3 = sig_3 + np.einsum("ja,jbiI->Iiab", v_ref1, tei_tmp) - np.einsum("jb,jaiI->Iiab", v_ref1, tei_tmp)
            #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_3 = sig_3 + np.einsum("ic,abIc->Iiab", v_ref1, tei_tmp_J) - np.einsum("ic,abcI->Iiab", v_ref1, tei_tmp_K)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 + 0.5*(np.einsum("Iicd,abcd->Iiab", v_ref3, tei_tmp) - np.einsum("Iicd,abdc->Iiab", v_ref3, tei_tmp))
            #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ijcb,ajci->Iiab", v_ref3, tei_tmp) + np.einsum("Ijca,bjci->Iiab", v_ref3, tei_tmp)
            #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_3 = sig_3 - (np.einsum("Jiac,JbIc->Iiab", v_ref3, tei_tmp_J) - np.einsum("Jiac,JbcI->Iiab", v_ref3, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_3 = sig_3 + (np.einsum("Jibc,JaIc->Iiab", v_ref3, tei_tmp_J) - np.einsum("Jibc,JacI->Iiab", v_ref3, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
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
        if(n_SF==1 and delta_ec==0 and conf_space=="p"):
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
            # v(2) indexing: (Ai:ab)
            v_ref2 = np.reshape(v_b2, (na_virt, socc))
            # v(3) unpack to indexing: (Aijb:aaab)
            v_ref3 = np.zeros((na_virt, socc, socc, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        for b in range(socc):
                            v_ref3[A, i, j, b] = v_b3[index]
                            v_ref3[A, j, i, b] = -1.0*v_b3[index]
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
            Fb_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Ai,aA->ia", v_ref2, Fb_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            sig_1 = sig_1 - np.einsum("Bj,ajBi->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 

            #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Ajia,jA->ia", v_ref3, Fa_tmp)
            #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - np.einsum("Aijb,ajbA->ia", v_ref3, tei_tmp)
            #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - 0.5*(np.einsum("Ajka,jkAi->ia", v_ref3, tei_tmp) - np.einsum("Ajka,jkiA->ia", v_ref3, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
            Fb_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_2 = np.einsum("ib,Ab->Ai", v_ref1, Fb_tmp)
            #   sig(iA:ab) += v(jb:ab)*t(jb:ab)*I(Ajbi:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_2 = sig_2 - np.einsum("jb,Ajbi->Ai", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 

            #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_2 = sig_2 + np.einsum("Bi,AB->Ai", v_ref2, Fb_tmp) - np.einsum("Aj,ji->Ai", v_ref2, Fa_tmp)
            #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_2 = sig_2 - np.einsum("Bj,AjBi->Ai", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_2 = sig_2 - np.einsum("Bijc,AjcB->Ai", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_3 = (np.einsum("jb,iA->Aijb", v_ref1, Fa_tmp) - np.einsum("ib,jA->Aijb", v_ref1, Fa_tmp))
            #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("ic,Abjc->Aijb", v_ref1, tei_tmp) + np.einsum("jc,Abic->Aijb", v_ref1, tei_tmp)
            #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_3 = sig_3 - (np.einsum("kb,Akij->Aijb", v_ref1, tei_tmp) - np.einsum("kb,Akji->Aijb", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 - np.einsum("Ci,AbjC->Aijb", v_ref2, tei_tmp)
            sig_3 = sig_3 + np.einsum("Cj,AbiC->Aijb", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
            F_bc_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            F_AB_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("Aijc,bc->Aijb", v_ref3, F_bc_tmp) # no contribution
            sig_3 = sig_3 + np.einsum("Bijb,AB->Aijb", v_ref3, F_AB_tmp) # no contribution
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Aikb,kj->Aijb", v_ref3, Fi_tmp) + np.einsum("Ajkb,ki->Aijb", v_ref3, Fi_tmp)
            #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_3 = sig_3 + np.einsum("Bijc,AbBc->Aijb", v_ref3, tei_tmp)
            #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 + 0.5*(np.einsum("Aklb,klij->Aijb", v_ref3, tei_tmp) - np.einsum("Aklb,klji->Aijb", v_ref3, tei_tmp))
            #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
            sig_3 = sig_3 - np.einsum("Akjc,kbic->Aijb", v_ref3, tei_tmp) 
            #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
            sig_3 = sig_3 + np.einsum("Akic,kbjc->Aijb", v_ref3, tei_tmp)
            #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
            tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 - (np.einsum("Cikb,AkCj->Aijb", v_ref3, tei_tmp_J) - np.einsum("Cikb,AkjC->Aijb", v_ref3, tei_tmp_K))
            #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
            sig_3 = sig_3 + (np.einsum("Cjkb,AkCi->Aijb", v_ref3, tei_tmp_J) - np.einsum("Cjkb,AkiC->Aijb", v_ref3, tei_tmp_K))

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
                    for A in range(na_virt):
                        for b in range(socc):
                            sig_3_out[index] = sig_3[A, i, j, b]  
                            index = index + 1 

            # combine and return
            return np.vstack((sig_1, sig_2, sig_3_out))

        # do excitation scheme: RAS(h,p)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="h,p"):
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
            n_b1_dets = int(socc*socc)
            n_b2_dets = int(nb_occ*socc)
            n_b3_dets = int(socc*na_virt)
            n_b4_dets = int(nb_occ*socc*(socc*(socc-1)/2))
            n_b5_dets = int(na_virt*socc*(socc*(socc-1)/2))
            v_b1 = v[0:n_b1_dets] # v for block 1
            v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets] # v for block 2
            v_b3 = v[n_b1_dets+n_b2_dets:n_b1_dets+n_b2_dets+n_b3_dets] # v for block 3
            v_b4 = v[n_b1_dets+n_b2_dets+n_b3_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets] # v for block 4
            v_b5 = v[n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets+n_b5_dets] # v for block 5
            # v(1) indexing: (ia:ab)
            v_ref1 = np.reshape(v_b1, (socc, socc))
            # v(2) indexing: (Ia:ab)
            v_ref2 = np.reshape(v_b2, (nb_occ, socc))
            # v(3) indexing: (Ai:ab)
            v_ref3 = np.reshape(v_b3, (na_virt, socc))
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
            # v(5) unpack to indexing: (Aijb:aaab)
            v_ref5 = np.zeros((na_virt, socc, socc, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        for b in range(socc):
                            v_ref5[A, i, j, b] = v_b5[index]
                            v_ref5[A, j, i, b] = -1.0*v_b5[index]
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("jb,ajbi->ia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 

            #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Ia,Ii->ia", v_ref2, Fa_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ib,Iaib->ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            
            #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
            Fb_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Ai,aA->ia", v_ref3, Fb_tmp)
            #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            sig_1 = sig_1 - np.einsum("Bj,ajBi->ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,4) v(4) = sig(1)
            ################################################ 

            #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_1 = sig_1 + np.einsum("Iiba,bI->ia", v_ref4, Fb_tmp)
            #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ijba,Ijbi->ia", v_ref4, tei_tmp)
            #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + 0.5*(np.einsum("Iicb,Iacb->ia", v_ref4, tei_tmp) - np.einsum("Iicb,Iabc->ia", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,5) v(5) = sig(1)
            ################################################ 

            #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Ajia,jA->ia", v_ref5, Fa_tmp)
            #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - np.einsum("Aijb,ajbA->ia", v_ref5, tei_tmp)
            #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - 0.5*(np.einsum("Ajka,jkAi->ia", v_ref5, tei_tmp) - np.einsum("Ajka,jkiA->ia", v_ref5, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
            Fa_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("ia,iI->Ia", v_ref1, Fa_tmp)
            #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
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
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_2 = sig_2 - np.einsum("Jb,JaIb->Ia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += -1.0*v(iA:ab)*I(aiAI:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 1)
            sig_2 = sig_2 - np.einsum("Ai,aiAI->Ia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,4) v(4) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += - v(Jiab:babb)*I(JibI:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 + np.einsum("Jiab,JibI->Ia", v_ref4, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,5) v(5) = sig(2)
            ################################################ 

            #   sig(Ia:ab) += - v(ijAa:aaab)*I(ijAI:aaaa)
            tei_tmp_J = self.tei.get_subblock(2, 2, 3, 1)
            tei_tmp_K = self.tei.get_subblock(2, 2, 1, 3)
            sig_2 = sig_2 - 0.5*(np.einsum("Aija,ijAI->Ia", v_ref5, tei_tmp_J) - np.einsum("Aija,ijIA->Ia", v_ref5, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 

            #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
            Fb_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_3 = np.einsum("ib,Ab->Ai", v_ref1, Fb_tmp)
            #   sig(iA:ab) += v(jb:ab)*I(Ajbi:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("jb,Ajbi->Ai", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 

            #   sig(iA:ab) += -v(Ia:ab)*I(AIai:baba)
            tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
            sig_3 = sig_3 - np.einsum("Ia,AIai->Ai", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 

            #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            Fb_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("Bi,AB->Ai", v_ref3, Fb_tmp) - np.einsum("Aj,ji->Ai", v_ref3, Fa_tmp)
            #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_3 = sig_3 - np.einsum("Bj,AjBi->Ai", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,4) v(4) = sig(3)
            ################################################ 

            #   sig(iA:ab) += v(Iibc:babb)*I(IAbc:bbbb)
            tei_tmp = self.tei.get_subblock(1, 3, 2, 2)
            sig_3 = sig_3 + 0.5*(np.einsum("Iibc,IAbc->Ai", v_ref4, tei_tmp) - np.einsum("Iibc,IAcb->Ai", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(3,5) v(5) = sig(3)
            ################################################ 

            #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 - np.einsum("Bijc,AjcB->Ai", v_ref5, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,1) v(1) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_4 = (np.einsum("ib,Ia->Iiab", v_ref1, Fb_tmp) - np.einsum("ia,Ib->Iiab", v_ref1, Fb_tmp))
            #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
            sig_4 = sig_4 + np.einsum("ja,jbiI->Iiab", v_ref1, tei_tmp) - np.einsum("jb,jaiI->Iiab", v_ref1, tei_tmp)
            #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_4 = sig_4 + np.einsum("ic,abIc->Iiab", v_ref1, tei_tmp_J) - np.einsum("ic,abcI->Iiab", v_ref1, tei_tmp_K)

            ################################################ 
            # Do the following term:
            #       H(4,2) v(2) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_4 = sig_4 + np.einsum("Ja,JbiI->Iiab", v_ref2, tei_tmp)
            sig_4 = sig_4 - np.einsum("Jb,JaiI->Iiab", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,3) v(3) = sig(4)
            ################################################ 

            #   sig(Iiab:babb) += v(iA:ab)*I(abIA:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 3)
            tei_tmp_K = self.tei.get_subblock(2, 2, 3, 1)
            sig_4 = sig_4 + (np.einsum("Ai,abIA->Iiab", v_ref3, tei_tmp_J) - np.einsum("Ai,abAI->Iiab", v_ref3, tei_tmp_K))

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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_4 = sig_4 + 0.5*(np.einsum("Iicd,abcd->Iiab", v_ref4, tei_tmp) - np.einsum("Iicd,abdc->Iiab", v_ref4, tei_tmp))
            #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_4 = sig_4 - np.einsum("Ijcb,ajci->Iiab", v_ref4, tei_tmp) + np.einsum("Ijca,bjci->Iiab", v_ref4, tei_tmp)
            #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_4 = sig_4 - (np.einsum("Jiac,JbIc->Iiab", v_ref4, tei_tmp_J) - np.einsum("Jiac,JbcI->Iiab", v_ref4, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_4 = sig_4 + (np.einsum("Jibc,JaIc->Iiab", v_ref4, tei_tmp_J) - np.einsum("Jibc,JacI->Iiab", v_ref4, tei_tmp_K))
            #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_4 = sig_4 + np.einsum("Jjab,JjIi->Iiab", v_ref4, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(4,5) v(5) = sig(4) (NC)
            ################################################ 

            #   sig(Iiab:babb) += v(jiAa:aaab)*I(jbAI:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 1)
            sig_4 = sig_4 - (np.einsum("Ajia,jbAI->Iiab", v_ref5, tei_tmp) - np.einsum("Ajib,jaAI->Iiab", v_ref5, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,1) v(1) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_5 = (np.einsum("jb,iA->Aijb", v_ref1, Fa_tmp) - np.einsum("ib,jA->Aijb", v_ref1, Fa_tmp))
            #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_5 = sig_5 - np.einsum("ic,Abjc->Aijb", v_ref1, tei_tmp) + np.einsum("jc,Abic->Aijb", v_ref1, tei_tmp)
            #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_5 = sig_5 - (np.einsum("kb,Akij->Aijb", v_ref1, tei_tmp) - np.einsum("kb,Akji->Aijb", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,2) v(2) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += -v(Ib:ab)*I(AIij:aaaa)
            tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
            sig_5 = sig_5 - (np.einsum("Ib,AIij->Aijb", v_ref2, tei_tmp) - np.einsum("Ib,AIji->Aijb", v_ref2, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,3) v(3) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_5 = sig_5 - np.einsum("Ci,AbjC->Aijb", v_ref3, tei_tmp)
            sig_5 = sig_5 + np.einsum("Cj,AbiC->Aijb", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(5,4) v(4) = sig(5)
            ################################################

            #   sig(ijAb:aaab) += - v(Iicb:babb)*I(Abjc:abab)
            tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
            sig_5 = sig_5 - (np.einsum("Iicb,AIjc->Aijb", v_ref4, tei_tmp) - np.einsum("Ijcb,AIic->Aijb", v_ref4, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(5,5) v(5) = sig(5)
            ################################################ 

            #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
            F_bc_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            F_AB_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_5 = sig_5 + np.einsum("Aijc,bc->Aijb", v_ref5, F_bc_tmp) # no contribution
            sig_5 = sig_5 + np.einsum("Bijb,AB->Aijb", v_ref5, F_AB_tmp) # no contribution
            Fi_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_5 = sig_5 - np.einsum("Aikb,kj->Aijb", v_ref5, Fi_tmp) + np.einsum("Ajkb,ki->Aijb", v_ref5, Fi_tmp)
            #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_5 = sig_5 + np.einsum("Bijc,AbBc->Aijb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_5 = sig_5 + 0.5*(np.einsum("Aklb,klij->Aijb", v_ref5, tei_tmp) - np.einsum("Aklb,klji->Aijb", v_ref5, tei_tmp))
            #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
            sig_5 = sig_5 - np.einsum("Akjc,kbic->Aijb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
            sig_5 = sig_5 + np.einsum("Akic,kbjc->Aijb", v_ref5, tei_tmp)
            #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
            tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
            sig_5 = sig_5 - (np.einsum("Cikb,AkCj->Aijb", v_ref5, tei_tmp_J) - np.einsum("Cikb,AkjC->Aijb", v_ref5, tei_tmp_K))
            #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
            sig_5 = sig_5 + (np.einsum("Cjkb,AkCi->Aijb", v_ref5, tei_tmp_J) - np.einsum("Cjkb,AkiC->Aijb", v_ref5, tei_tmp_K))

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
                    for A in range(na_virt):
                        for b in range(socc):
                            sig_5_out[index] = sig_5[A, i, j, b]
                            index = index + 1

            # combine and return
            return np.vstack((sig_1, sig_2, sig_3, sig_4_out, sig_5_out))

        # doing EA calculation
        if(n_SF==0 and delta_ec==1 and conf_space==""):
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

        # doing RAS(h)-EA calculation
        if(n_SF==0 and delta_ec==1 and conf_space=="h"):
            """  
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(a:b)
                block2 = v(Iab:bbb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) + | H(1,2) | * v(2) = sig(1)
                | H(2,1) | * v(1) + | H(2,2) | * v(2) = sig(2)
           
            """
            v_b1 = v[0:socc] # v for block 1
            v_b2 = v[socc:] # v for block 2
            # v(1) indexing: (a:b)
            v_ref1 = np.reshape(v_b1, (socc))
            # v(2) indexing: (Iab:bbb)
            v_ref2 = np.zeros((nb_occ, socc, socc))
            index = 0
            for I in range(nb_occ):
                for a in range(socc):
                    for b in range(a):
                        v_ref2[I, a, b] = v_b2[index]
                        v_ref2[I, b, a] = -1.0*v_b2[index]
                        index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(a:b) += v(a:b)*F(ab:bb)
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("b,ab->a", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(a:b) += v(a:b)*F(ab:bb)
            F_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("Iba,Ib->a", v_ref2, F_tmp)
            #   sig(a:b) += 0.5*v(Ibc:bbb)*I(Iabc:bbbb)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + 0.5*(np.einsum("Ibc,Iabc->a", v_ref2, tei_tmp) - np.einsum("Ibc,Iacb->a", v_ref2, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            #   sig(Iab:bbb) += P(ab)*v(b:b)*F(aI:bb)
            F_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_2 = np.einsum("b,aI->Iab", v_ref1, F_tmp)
            sig_2 = sig_2 - np.einsum("a,bI->Iab", v_ref1, F_tmp) #P(ab)
            #   sig(Iab:bbb) += v(c:b)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_2 = sig_2 + (np.einsum("c,abIc->Iab", v_ref1, tei_tmp_J) - np.einsum("c,abcI->Iab", v_ref1, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            #   sig(Iab:bbb) += -v(Jab:bbb)*F(JI:bb)
            F_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_2 = sig_2 - np.einsum("Jab,JI->Iab", v_ref2, F_tmp)
            #   sig(Iab:bbb) += P(ab)*v(Icb:bbb)*F(ac:bb)
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("Icb,ac->Iab", v_ref2, F_tmp)
            sig_2 = sig_2 - np.einsum("Ica,bc->Iab", v_ref2, F_tmp) #P(ab)
            #   sig(Iab:bbb) += -P(ab)*v(Jac:bbb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 - (np.einsum("Jac,JbIc->Iab", v_ref2, tei_tmp_J) - np.einsum("Jac,JbcI->Iab", v_ref2, tei_tmp_K))
            sig_2 = sig_2 + (np.einsum("Jbc,JaIc->Iab", v_ref2, tei_tmp_J) - np.einsum("Jbc,JacI->Iab", v_ref2, tei_tmp_K)) #P(ab)
            #   sig(Iab:bbb) += 0.5*v(Icd:bbb)*I(abcd:bbbb)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_2 = sig_2 + 0.5*(np.einsum("Icd,abcd->Iab", v_ref2, tei_tmp) - np.einsum("Icd,abdc->Iab", v_ref2, tei_tmp))

            # Sigma evaluations done! Pack back up for returning
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2_out = np.zeros((v_b2.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for a in range(socc):
                    for b in range(a):
                        sig_2_out[index] = sig_2[I, a, b]
                        index = index + 1

            return np.vstack((sig_1, sig_2_out))

        # doing RAS(p)-EA calculation
        if(n_SF==0 and delta_ec==1 and conf_space=="p"):
            """  
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(a:b)
                block2 = v(A:b)
                block3 = v(iAa:aab)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) + | H(1,2) | * v(2) + | H(1,3) | * v(3) = sig(1)
                | H(2,1) | * v(1) + | H(2,2) | * v(2) + | H(2,3) | * v(3) = sig(2)
                | H(3,1) | * v(1) + | H(3,2) | * v(2) + | H(3,3) | * v(3) = sig(2)
           
            """
            v_b1 = v[0:socc] # v for block 1
            v_b2 = v[socc:socc+na_virt] # v for block 2
            v_b3 = v[socc+na_virt:] # v for block 3
            # v(1) indexing: (a:b)
            v_ref1 = np.reshape(v_b1, (socc))
            # v(2) indexing: (A:b)
            v_ref2 = np.reshape(v_b2, (na_virt))
            # v(3) indexing: (iAa:aab)
            v_ref3 = np.reshape(v_b3, (na_virt, socc, socc))

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = np.einsum("b,ab->a", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            F_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("A,aA->a", v_ref2, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            F_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Aia,iA->a", v_ref3, F_tmp)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            sig_1 = sig_1 + np.einsum("Aib,iaAb->a", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            F_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_2 = np.einsum("a,Aa->A", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            F_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_2 = sig_2 + np.einsum("B,AB->A", v_ref2, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 
            tei_tmp = self.tei.get_subblock(2, 3, 3, 2)
            sig_2 = sig_2 + np.einsum("Bia,iABa->A", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 
            F_tmp = Fa[na_occ:nbf, nb_occ:na_occ]
            sig_3 = np.einsum("a,Ai->Aia", v_ref1, F_tmp)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_3 = sig_3 + np.einsum("b,Aaib->Aia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 + np.einsum("B,AaiB->Aia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Aja,ji->Aia", v_ref3, F_tmp)
            F_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("Bia,AB->Aia", v_ref3, F_tmp)
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Aib,ab->Aia", v_ref3, F_tmp)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ajb,jaib->Aia", v_ref3, tei_tmp)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_3 = sig_3 + np.einsum("Bib,AaBb->Aia", v_ref3, tei_tmp)
            tei_tmp_J = self.tei.get_subblock(3, 2, 2, 3)
            tei_tmp_K = self.tei.get_subblock(3, 2, 3, 2)
            sig_3 = sig_3 + (np.einsum("Bja,AjiB->Aia", v_ref3, tei_tmp_J) - np.einsum("Bja,AjBi->Aia", v_ref3, tei_tmp_K))

            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            sig_3 = np.reshape(sig_3, (v_b3.shape[0], 1))

            return np.vstack((sig_1, sig_2, sig_3))

        # doing IP calculation
        if(n_SF==0 and delta_ec==-1 and conf_space==""):
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

        # doing RAS(h)-IP calculation
        if(n_SF==0 and delta_ec==-1 and conf_space=="h"):
            """  
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(i:a)
                block2 = v(I:a)
                block3 = v(Iia:bab)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) + | H(1,2) | * v(2) + | H(1,3) | * v(3) = sig(1)
                | H(2,1) | * v(1) + | H(2,2) | * v(2) + | H(2,3) | * v(3) = sig(2)
                | H(3,1) | * v(1) + | H(3,2) | * v(2) + | H(3,3) | * v(3) = sig(3)
           
            """
            v_b1 = v[0:socc] # v for block 1
            v_b2 = v[socc:nb_occ+socc] # v for block 2
            v_b3 = v[nb_occ+socc:] # v for block 3
            # v(1) indexing: (i:a)
            v_ref1 = np.reshape(v_b1, (socc))
            # v(2) indexing: (I:a)
            v_ref2 = np.reshape(v_b2, (nb_occ))
            # v(3) indexing: (Iia:bba)
            v_ref3 = np.reshape(v_b3, (nb_occ, socc, socc))

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*np.einsum("j,ji->i", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            F_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("I,Ii->i", v_ref2, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            F_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("Iia,Ia->i", v_ref3, F_tmp)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ija,Ijai->i", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            F_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("i,iI->I", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            F_tmp = Fa[0:nb_occ, 0:nb_occ]
            sig_2 = sig_2 - np.einsum("J,JI->I", v_ref2, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 - np.einsum("Jia,JiaI->I", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 
            F_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_3 = np.einsum("i,aI->Iia", v_ref1, F_tmp)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
            sig_3 = sig_3 - np.einsum("j,ajIi->Iia", v_ref1, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################
            tei_tmp = self.tei.get_subblock(2, 1, 1, 2)
            sig_3 = sig_3 - np.einsum("J,aJIi->Iia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Ija,ji->Iia", v_ref3, F_tmp)
            F_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Iib,ab->Iia", v_ref3, F_tmp)
            F_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_3 = sig_3 - np.einsum("Jia,JI->Iia", v_ref3, F_tmp)

            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ijb,ajbi->Iia", v_ref3, tei_tmp)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_3 = sig_3 + np.einsum("Jja,JjIi->Iia", v_ref3, tei_tmp)
            tei_tmp_J = self.tei.get_subblock(2, 1, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 1, 2, 1)
            sig_3 = sig_3 + (np.einsum("Jib,aJIb->Iia", v_ref3, tei_tmp_J) - np.einsum("Jib,aJbI->Iia", v_ref3, tei_tmp_K))

            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2 = np.reshape(sig_2, (v_b2.shape[0], 1))
            sig_3 = np.reshape(sig_3, (v_b3.shape[0], 1))

            return np.vstack((sig_1, sig_2, sig_3))

        # doing RAS(p)-IP calculation
        if(n_SF==0 and delta_ec==-1 and conf_space=="p"):
            """  
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(i:a)
                block2 = v(ijA:aaa)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) + | H(1,2) | * v(2) = sig(1)
                | H(2,1) | * v(1) + | H(2,2) | * v(2) = sig(2)
           
            """
            v_b1 = v[0:socc] # v for block 1
            v_b2 = v[socc:] # v for block 2
            # v(1) indexing: (i:a)
            v_ref1 = np.reshape(v_b1, (socc))
            # v(2) indexing: (Aij:aaa)
            v_ref2 = np.zeros((na_virt, socc, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        v_ref2[A, i, j] = v_b2[index]
                        v_ref2[A, j, i] = -1.0*v_b2[index]
                        index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(i:a) += -v(j:a)*F(ji:aa)
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*np.einsum("j,ji->i", v_ref1, F_tmp)

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            #   sig(i:a) += v(jiA:aaa)*F(jA:aa)
            F_tmp = Fa[nb_occ:na_occ, na_occ:na_occ+na_virt]
            sig_1 = sig_1 + np.einsum("Aji,jA->i", v_ref2, F_tmp)
            #   sig(i:a) += -0.5*v(jkA:aaa)*I(jkAi:aaaa)
            tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - 0.5*(np.einsum("Ajk,jkAi->i", v_ref2, tei_tmp_J) - np.einsum("Ajk,jkiA->i", v_ref2, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            #   sig(ijA:aaa) += P(ij)*v(j:a)*F(Ai:aa)
            F_tmp = Fa[na_occ:na_occ+na_virt, nb_occ:na_occ]
            sig_2 = np.einsum("j,Ai->Aij", v_ref1, F_tmp)
            sig_2 = sig_2 - np.einsum("i,Aj->Aij", v_ref1, F_tmp) #P(ij)
            #   sig(ijA:aaa) += -v(k:a)*I(Akij:aaaa)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_2 = sig_2 - (np.einsum("k,Akij->Aij", v_ref1, tei_tmp) - np.einsum("k,Akji->Aij", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            #   sig(ijA:aaa) += v(ijB:aaa)*F(AB:aa)
            F_tmp = Fa[na_occ:na_occ+na_virt, na_occ:na_occ+na_virt]
            sig_2 = sig_2 + np.einsum("Bij,AB->Aij", v_ref2, F_tmp)
            #   sig(ijA:aaa) += -P(ij)*v(kjA:aaa)*F(ki:aa)
            F_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 - np.einsum("Akj,ki->Aij", v_ref2, F_tmp)
            sig_2 = sig_2 + np.einsum("Aki,kj->Aij", v_ref2, F_tmp) #P(ij)
            #   sig(ijA:aaa) += 0.5*v(klA:aaa)*I(klij:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_2 = sig_2 + 0.5*(np.einsum("Akl,klij->Aij", v_ref2, tei_tmp) - np.einsum("Akl,klji->Aij", v_ref2, tei_tmp))
            #   sig(ijA:aaa) += -P(ij)*v(ikB:aaa)*I(AkBj:aaaa)
            tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
            sig_2 = sig_2 - (np.einsum("Bik,AkBj->Aij", v_ref2, tei_tmp_J) - np.einsum("Bik,AkjB->Aij", v_ref2, tei_tmp_K))
            sig_2 = sig_2 + (np.einsum("Bjk,AkBi->Aij", v_ref2, tei_tmp_J) - np.einsum("Bjk,AkiB->Aij", v_ref2, tei_tmp_K)) #P(ij)

            # Sigma evaluations done! Pack back up for returning
            sig_1 = np.reshape(sig_1, (v_b1.shape[0], 1))
            sig_2_out = np.zeros((v_b2.shape[0], 1))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        sig_2_out[index] = sig_2[A, i, j]
                        index = index + 1

            return np.vstack((sig_1, sig_2_out))


        # do excitation scheme: 1SF-CAS-EA
        if(n_SF==1 and delta_ec==1 and conf_space==""):
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
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

        # do excitation scheme: CAS-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space==""):
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
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
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

        # do excitation scheme: RAS(h)-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space=="h"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(ija:aab)
                block2 = v(Iia:aab)
                block2 = v(Iijab:baabb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) + | H(1,2) | * v(2) + | H(1,3) | * v(3) = sig(1)
                | H(2,1) | * v(1) + | H(2,2) | * v(2) + | H(2,3) | * v(3) = sig(2)
                | H(3,1) | * v(1) + | H(3,2) | * v(2) + | H(3,3) | * v(3) = sig(3)
           
            """

            n_b1_dets = int(socc * ((socc-1)*(socc)/2))
            n_b2_dets = int(socc * nb_occ * socc)
            v_b1 = v[0:n_b1_dets]
            v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets]
            v_b3 = v[n_b1_dets+n_b2_dets:]

            # v(1) unpack to indexing: (ija:aab)
            v_ref1 = np.zeros((socc,socc,socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        v_ref1[i, j, a] = v_b1[index]
                        v_ref1[j, i, a] = -1.0*v_b1[index]
                        index = index + 1
            # v(2) unpack to indexing: (Iia:aab)
            v_ref2 = np.reshape(v_b2, (nb_occ, socc, socc))
            # v(3) unpack to indexing: (Iijab:aaabb)
            v_ref3 = np.zeros((nb_occ, socc, socc, socc, socc))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for j in range(i):
                        for a in range(socc):
                            for b in range(a):
                                v_ref3[I, i, j, a, b] = v_b3[index]
                                v_ref3[I, j, i, a, b] = -1.0*v_b3[index]
                                v_ref3[I, i, j, b, a] = -1.0*v_b3[index]
                                v_ref3[I, j, i, b, a] = v_b3[index]
                                index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(ija:aab) += -P(ij)*v(kja:aab)*F(ki:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = -1.0*np.einsum("kja,ki->ija", v_ref1, Fa_tmp)
            sig_1 = sig_1 + np.einsum("kia,kj->ija", v_ref1, Fa_tmp) #P(ij)
            #   sig(ija:aab) += v(ijb:aab)*F(ab:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("ijb,ab->ija", v_ref1, Fb_tmp)
            #   sig(ija:aab) += -P(ij)*v(ikb:aab)*I(akbj:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("ikb,akbj->ija", v_ref1, tei_tmp)
            sig_1 = sig_1 + np.einsum("jkb,akbi->ija", v_ref1, tei_tmp) #P(ij)
            #   sig(ija:aab) += 0.5*v(kla:aab)*I(klij:aaaa)
            sig_1 = sig_1 + 0.5*(np.einsum("kla,klij->ija", v_ref1, tei_tmp) - np.einsum("kla,klji->ija", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            #   sig(ija:aab) += -P(ij)*v(Ija:aab)*F(Ii:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Ija,Ii->ija", v_ref2, Fa_tmp)
            sig_1 = sig_1 + np.einsum("Iia,Ij->ija", v_ref2, Fa_tmp) #P(ij)
            #   sig(ija:aab) += v(Ika:aab)*F(Ikij:aaaa)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + (np.einsum("Ika,Ikij->ija", v_ref2, tei_tmp) - np.einsum("Ika,Ikji->ija", v_ref2, tei_tmp))
            #   sig(ija:aab) += -P(ij)*v(Ijb:aab)*F(aIbi:baba)
            tei_tmp = self.tei.get_subblock(2, 1, 2, 2)
            sig_1 = sig_1 - np.einsum("Ijb,aIbi->ija", v_ref2, tei_tmp)
            sig_1 = sig_1 + np.einsum("Iib,aIbj->ija", v_ref2, tei_tmp) #P(ij)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            #   sig(Iia:aab) += v(Iijba:baabb)*F(Ib:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Iijba,Ib->ija", v_ref3, Fb_tmp)
            #   sig(Iia:aab) += v(Iijbc:baabb)*F(Iabc:bbbb)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + 0.5*(np.einsum("Iijbc,Iabc->ija", v_ref3, tei_tmp) - np.einsum("Iijbc,Iacb->ija", v_ref3, tei_tmp))
            #   sig(Iia:aab) += -P(ij)*v(Ikjba:baabb)*F(Ikbi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - np.einsum("Ikjba,Ikbi->ija", v_ref3, tei_tmp)
            sig_1 = sig_1 + np.einsum("Ikiba,Ikbj->ija", v_ref3, tei_tmp) #P(ij)

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            #   sig(Iia:aab) += -1.0*v(jia:aab)*F(Ij:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_2 = -1.0*np.einsum("jia,Ij->Iia", v_ref1, Fa_tmp)
            #   sig(ija:aab) += -1.0*v(jib:aab)*I(ajbI:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
            sig_2 = sig_2 - np.einsum("jib,ajbI->Iia", v_ref1, tei_tmp) 
            #   sig(ija:aab) += 0.5*v(jka:aab)*I(jkIi:aaaa)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_2 = sig_2 + 0.5*(np.einsum("jka,jkIi->Iia", v_ref1, tei_tmp_J) - np.einsum("jka,jkiI->Iia", v_ref1, tei_tmp_K))

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            #   sig(Iia:aab) += -1.0*v(Jia:aab)*F(JI:aa)
            Fa_tmp = Fa[0:nb_occ, 0:nb_occ]
            sig_2 = sig_2 - np.einsum("Jia,JI->Iia", v_ref2, Fa_tmp)
            #   sig(Iia:aab) += -1.0*v(Jia:aab)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 - np.einsum("Ija,ji->Iia", v_ref2, Fa_tmp)
            #   sig(Iia:aab) += v(Iib:aab)*F(ab:aa)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("Iib,ab->Iia", v_ref2, Fb_tmp)
            #   sig(Iia:aab) += v(Jja:aab)*I(JjIi:aaaa)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 + (np.einsum("Jja,JjIi->Iia", v_ref2, tei_tmp_J) - np.einsum("Jja,JjiI->Iia", v_ref2, tei_tmp_K))
            #   sig(Iia:aab) += -1.0*v(Ijb:aab)*I(ajbi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_2 = sig_2 - np.einsum("Ijb,ajbi->Iia", v_ref2, tei_tmp)
            #   sig(Iia:aab) += -1.0*v(Jib:aab)*I(aJbI:baba)
            tei_tmp = self.tei.get_subblock(2, 1, 2, 1)
            sig_2 = sig_2 - np.einsum("Jib,aJbI->Iia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 
            #   sig(Iia:aab) += -1.0*v(Jjiba:baabb)*F(JjbI:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 - np.einsum("Jjiba,JjbI->Iia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 
            #   sig(Iijab:baabb) += P(ab)*v(ijb:aab)*F(aI:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_3 = np.einsum("ijb,aI->Iijab", v_ref1, Fb_tmp)
            sig_3 = sig_3 - np.einsum("ija,bI->Iijab", v_ref1, Fb_tmp) #P(ab)
            #   sig(Iijab:baabb) += v(ijc:aab)*I(abIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_3 = sig_3 + (np.einsum("ijc,abIc->Iijab", v_ref1, tei_tmp_J) - np.einsum("ijc,abcI->Iijab", v_ref1, tei_tmp_K))
            #   sig(Iijab:baabb) += -1.0*P(ij)*P(ab)*v(kjb:aab)*I(akIi:bbbb)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
            sig_3 = sig_3 - np.einsum("kjb,akIi->Iijab", v_ref1, tei_tmp)
            sig_3 = sig_3 + np.einsum("kib,akIj->Iijab", v_ref1, tei_tmp) #P(ij)
            sig_3 = sig_3 + np.einsum("kja,bkIi->Iijab", v_ref1, tei_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("kia,bkIj->Iijab", v_ref1, tei_tmp) #P(ij)P(ab)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 
            #   sig(Iijab:baabb) += -1.0*P(ij)*v(Jjb:aab)*I(aJIi:baba)
            tei_tmp = self.tei.get_subblock(2, 1, 1, 2)
            sig_3 = sig_3 - np.einsum("Jjb,aJIi->Iijab", v_ref2, tei_tmp)
            sig_3 = sig_3 + np.einsum("Jib,aJIj->Iijab", v_ref2, tei_tmp) #P(ij)
            sig_3 = sig_3 + np.einsum("Jja,bJIi->Iijab", v_ref2, tei_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("Jia,bJIj->Iijab", v_ref2, tei_tmp) #P(ij)P(ab)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 
            #   sig(Iijab:baabb) += P(ab)*v(Iijcb:baabb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Iijcb,ac->Iijab", v_ref3, Fb_tmp)
            sig_3 = sig_3 - np.einsum("Iijca,bc->Iijab", v_ref3, Fb_tmp) #P(ab)
            #   sig(Iijab:baabb) += -1.0*v(Jijab:baabb)*F(JI:bb)
            Fb_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_3 = sig_3 - np.einsum("Jijab,JI->Iijab", v_ref3, Fb_tmp)
            #   sig(Iijab:baabb) += -P(ij)*v(Ikjab:baabb)*F(ki:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Ikjab,ki->Iijab", v_ref3, Fa_tmp)
            sig_3 = sig_3 + np.einsum("Ikiab,kj->Iijab", v_ref3, Fa_tmp) #P(ij)
            #   sig(Iijab:baabb) += P(ij)*v(Jkjab:baabb)*I(JkIi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_3 = sig_3 + np.einsum("Jkjab,JkIi->Iijab", v_ref3, tei_tmp)
            sig_3 = sig_3 - np.einsum("Jkiab,JkIj->Iijab", v_ref3, tei_tmp) #P(ij)
            #   sig(Iijab:baabb) += -P(ab)*v(Jijac:baabb)*I(JbIc:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_3 = sig_3 - (np.einsum("Jijac,JbIc->Iijab", v_ref3, tei_tmp_J) - np.einsum("Jijac,JbcI->Iijab", v_ref3, tei_tmp_K))
            sig_3 = sig_3 + (np.einsum("Jijbc,JaIc->Iijab", v_ref3, tei_tmp_J) - np.einsum("Jijbc,JacI->Iijab", v_ref3, tei_tmp_K)) #P(ab)
            #   sig(Iijab:baabb) += -P(ab)*P(ij)*v(Ikjcb:baabb)*I(ciak:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ikjcb,ciak->Iijab", v_ref3, tei_tmp)
            sig_3 = sig_3 + np.einsum("Ikicb,cjak->Iijab", v_ref3, tei_tmp) #P(ij)
            sig_3 = sig_3 + np.einsum("Ikjca,cibk->Iijab", v_ref3, tei_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("Ikica,cjbk->Iijab", v_ref3, tei_tmp) #P(ij)P(ab)
            #   sig(Iijab:baabb) += 0.5*v(Iijcd:baabb)*I(abcd:bbbb)
            sig_3 = sig_3 + 0.5*(np.einsum("Iijcd,abcd->Iijab", v_ref3, tei_tmp) - np.einsum("Iijcd,abdc->Iijab", v_ref3, tei_tmp))
            #   sig(Iijab:baabb) += 0.5*v(Iklab:baabb)*I(klij:bbbb)
            sig_3 = sig_3 + 0.5*(np.einsum("Iklab,klij->Iijab", v_ref3, tei_tmp) - np.einsum("Iklab,klji->Iijab", v_ref3, tei_tmp))

            sig_1_out = np.zeros((v_b1.shape[0], 1))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        sig_1_out[index] = sig_1[i, j, a]
                        index = index + 1
            # v(2) repack
            sig_2_out = np.reshape(sig_2, (v_b2.shape[0], 1))
            # v(3) repack
            sig_3_out = np.zeros((v_b3.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for j in range(i):
                        for a in range(socc):
                            for b in range(a):
                                sig_3_out[index] = sig_3[I, i, j, a, b]
                                index = index + 1

            return np.vstack((sig_1_out, sig_2_out, sig_3_out))

        # do excitation scheme: RAS(h)-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space=="h"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(iab:abb)
                block2 = v(Iab:abb)
                block2 = v(Iiabc:babbb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """
            n_b1_dets = int(socc * ((socc-1)*(socc)/2)) 
            n_b2_dets = int(nb_occ * ((socc-1)*(socc)/2))
            v_b1 = v[0:n_b1_dets]
            v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets]
            v_b3 = v[n_b1_dets+n_b2_dets:]
            # v(1) unpack to indexing: (iab:abb)
            v_ref1 = np.zeros((socc,socc,socc))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref1[i, a, b] = v_b1[index]
                        v_ref1[i, b, a] = -1.0*v_b1[index]
                        index = index + 1
            # v(2) unpack to indexing: (Iab:abb)
            v_ref2 = np.zeros((nb_occ,socc,socc))
            index = 0
            for I in range(nb_occ):
                for a in range(socc):
                    for b in range(a):
                        v_ref2[I, a, b] = v_b2[index]
                        v_ref2[I, b, a] = -1.0*v_b2[index]
                        index = index + 1
            # v(3) unpack to indexing: (Iiabc:babbb)
            v_ref3 = np.zeros((nb_occ,socc,socc,socc,socc))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            for c in range(b):
                                v_ref3[I, i, a, b, c] = v_b3[index]
                                v_ref3[I, i, b, a, c] = -1.0*v_b3[index]
                                v_ref3[I, i, b, c, a] = v_b3[index]
                                v_ref3[I, i, c, b, a] = -1.0*v_b3[index]
                                v_ref3[I, i, c, a, b] = v_b3[index]
                                v_ref3[I, i, a, c, b] = -1.0*v_b3[index]
                                index = index + 1

            ################################################ 
            # Do the following term:
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(ija:aab) += P(ab)*t(iab:abb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = (np.einsum("icb,ac->iab", v_ref1, Fb_tmp) - np.einsum("ica,bc->iab", v_ref1, Fb_tmp))
            #   sig(ija:aab) += -t(jab:abb)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("jab,ji->iab", v_ref1, Fa_tmp)
            #   sig(ija:aab) += -P(ab)*t(jac:abb)*I(jbic:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - (np.einsum("jac,jbic->iab", v_ref1, tei_tmp) - np.einsum("jbc,jaic->iab", v_ref1, tei_tmp))
            #   sig(ija:aab) += 0.5*t(icd:abb)*I(abcd:abab)
            sig_1 = sig_1 + 0.5*(np.einsum("icd,abcd->iab", v_ref1, tei_tmp) - np.einsum("icd,abdc->iab", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            #   sig(iab:abb) += -t(Iab:abb)*F(Ii:aa)
            Fa_tmp = Fa[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("Iab,Ii->iab", v_ref2, Fa_tmp)
            #   sig(iab:abb) += -P(ab)*t(Iac:abb)*I(Ibic:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 - (np.einsum("Iac,Ibic->iab", v_ref2, tei_tmp) - np.einsum("Ibc,Iaic->iab", v_ref2, tei_tmp))
            # for testing!
            sig_1_2 = -1.0*np.einsum("Iab,Ii->iab", v_ref2, Fa_tmp)
            sig_1_2 = sig_1_2 - (np.einsum("Iac,Ibic->iab", v_ref2, tei_tmp) - np.einsum("Ibc,Iaic->iab", v_ref2, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            #   sig(iab:abb) += t(Iicab:babbb)*F(Ic:bb)
            Fb_tmp = Fb[0:nb_occ, nb_occ:na_occ]
            sig_1 = sig_1 + np.einsum("Iicab,Ic->iab", v_ref3, Fb_tmp)
            #   sig(iab:abb) += -t(Ijacb:abb)*I(Ijci:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
            sig_1 = sig_1 + np.einsum("Ijacb,Ijci->iab", v_ref3, tei_tmp)
            sig_1 = sig_1 + (np.einsum("Iicdb,Idca->iab", v_ref3, tei_tmp) - np.einsum("Iicdb,Idac->iab", v_ref3, tei_tmp))
            sig_1 = sig_1 - (np.einsum("Iicda,Idcb->iab", v_ref3, tei_tmp) - np.einsum("Iicda,Idbc->iab", v_ref3, tei_tmp)) #P(ab)
            # for testing!
            sig_1_3 = np.einsum("Iicab,Ic->iab", v_ref3, Fb_tmp)
            sig_1_3 = sig_1_3 + np.einsum("Ijacb,Ijci->iab", v_ref3, tei_tmp)
            sig_1_3 = sig_1_3 + (np.einsum("Iicdb,Idca->iab", v_ref3, tei_tmp) - np.einsum("Iicdb,Idac->iab", v_ref3, tei_tmp))
            sig_1_3 = sig_1_3 - (np.einsum("Iicda,Idcb->iab", v_ref3, tei_tmp) - np.einsum("Iicda,Idbc->iab", v_ref3, tei_tmp)) #P(ab)

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            #   sig(Iab:abb) += -t(iab:abb)*F(iI:aa)
            Fa_tmp = Fa[nb_occ:na_occ, 0:nb_occ]
            sig_2 = -1.0*np.einsum("iab,iI->Iab", v_ref1, Fa_tmp)
            #   sig(Iab:abb) += -P(ab)*t(iac:abb)*I(ibIc:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
            sig_2 = sig_2 - (np.einsum("iac,ibIc->Iab", v_ref1, tei_tmp) - np.einsum("ibc,iaIc->Iab", v_ref1, tei_tmp))
            # for testing!
            sig_2_1 = -1.0*np.einsum("iab,iI->Iab", v_ref1, Fa_tmp)
            sig_2_1 = sig_2_1 - (np.einsum("iac,ibIc->Iab", v_ref1, tei_tmp) - np.einsum("ibc,iaIc->Iab", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            #   sig(Iab:abb) += P(ab)*t(Icb:abb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + (np.einsum("Icb,ac->Iab", v_ref2, Fb_tmp) - np.einsum("Ica,bc->Iab", v_ref2, Fb_tmp))
            #   sig(Iab:abb) += -t(Jab:abb)*F(JI:aa)
            Fa_tmp = Fa[0:nb_occ, 0:nb_occ]
            sig_2 = sig_2 - np.einsum("Jab,JI->Iab", v_ref2, Fa_tmp)
            #   sig(Iab:abb) += -P(ab)*t(Jac:abb)*I(JbIc:abab)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_2 = sig_2 - (np.einsum("Jac,JbIc->Iab", v_ref2, tei_tmp) - np.einsum("Jbc,JaIc->Iab", v_ref2, tei_tmp))
            #   sig(Iab:abb) += 0.5*t(Icd:abb)*I(abcd:bbbb)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_2 = sig_2 + 0.5*(np.einsum("Icd,abcd->Iab", v_ref2, tei_tmp) - np.einsum("Icd,abdc->Iab", v_ref2, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 
            #   sig(Iab:abb) += t(Jiacb:babbb)*I(JicI:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
            sig_2 = sig_2 + np.einsum("Jiacb,JicI->Iab", v_ref3, tei_tmp)
            # for testing
            sig_2_3 = np.einsum("Jiacb,JicI->Iab", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 
            #   sig(Iiabc:babbb) += P(ab)*P(ac)*t(ibc:abb)*F(aI:bb)
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_3 = np.einsum("ibc,aI->Iiabc", v_ref1, Fb_tmp)
            sig_3 = sig_3 - np.einsum("iac,bI->Iiabc", v_ref1, Fb_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("iba,cI->Iiabc", v_ref1, Fb_tmp) #P(ac)
            sig_3 = sig_3 + np.einsum("ica,bI->Iiabc", v_ref1, Fb_tmp) #P(ab)P(ac)
            #   sig(Iiabc:babbb) += P(ac)*P(bc)*t(idc:abb)*I(abId:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_3 = sig_3 + (np.einsum("idc,abId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idc,abdI->Iiabc", v_ref1, tei_tmp_K))
            sig_3 = sig_3 - (np.einsum("ida,cbId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("ida,cbdI->Iiabc", v_ref1, tei_tmp_K)) # P(ac)
            sig_3 = sig_3 - (np.einsum("idb,acId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idb,acdI->Iiabc", v_ref1, tei_tmp_K)) # P(bc)
            sig_3 = sig_3 + (np.einsum("idb,caId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idb,cadI->Iiabc", v_ref1, tei_tmp_K)) # P(bc)P(ac)
            #   sig(Iiabc:babbb) += -P(ac)*P(ab)*t(jbc:abb)*I(ajIi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
            sig_3 = sig_3 - np.einsum("jbc,ajIi->Iiabc", v_ref1, tei_tmp)
            sig_3 = sig_3 + np.einsum("jac,bjIi->Iiabc", v_ref1, tei_tmp) # P(ab)
            sig_3 = sig_3 + np.einsum("jba,cjIi->Iiabc", v_ref1, tei_tmp) # P(ac)
            sig_3 = sig_3 - np.einsum("jca,bjIi->Iiabc", v_ref1, tei_tmp) # P(ab) P(ac)

            # for testing
            Fb_tmp = Fb[nb_occ:na_occ, 0:nb_occ]
            sig_3_1 = np.einsum("ibc,aI->Iiabc", v_ref1, Fb_tmp)
            sig_3_1 = sig_3_1 - np.einsum("iac,bI->Iiabc", v_ref1, Fb_tmp) #P(ab)
            sig_3_1 = sig_3_1 - np.einsum("iba,cI->Iiabc", v_ref1, Fb_tmp) #P(ac)
            sig_3_1 = sig_3_1 + np.einsum("ica,bI->Iiabc", v_ref1, Fb_tmp) #P(ab)P(ac)
            #   sig(Iiabc:babbb) += P(ac)*P(bc)*t(idc:abb)*I(abId:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
            sig_3_1 = sig_3_1 + (np.einsum("idc,abId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idc,abdI->Iiabc", v_ref1, tei_tmp_K))
            sig_3_1 = sig_3_1 - (np.einsum("ida,cbId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("ida,cbdI->Iiabc", v_ref1, tei_tmp_K)) # P(ac)
            sig_3_1 = sig_3_1 - (np.einsum("idb,acId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idb,acdI->Iiabc", v_ref1, tei_tmp_K)) # P(bc)
            sig_3_1 = sig_3_1 + (np.einsum("idb,caId->Iiabc", v_ref1, tei_tmp_J) - np.einsum("idb,cadI->Iiabc", v_ref1, tei_tmp_K)) # P(bc)P(ac)
            #   sig(Iiabc:babbb) += -P(ac)*P(ab)*t(jbc:abb)*I(ajIi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
            sig_3_1 = sig_3_1 - np.einsum("jbc,ajIi->Iiabc", v_ref1, tei_tmp)
            sig_3_1 = sig_3_1 + np.einsum("jac,bjIi->Iiabc", v_ref1, tei_tmp) # P(ab)
            sig_3_1 = sig_3_1 + np.einsum("jba,cjIi->Iiabc", v_ref1, tei_tmp) # P(ac)
            sig_3_1 = sig_3_1 - np.einsum("jca,bjIi->Iiabc", v_ref1, tei_tmp) # P(ab) P(ac)

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 
            #   sig(Iiabc:babbb) += -P(ab)*P(ac)*t(Jbc:abb)*F(aJIi:baba)
            tei_tmp = self.tei.get_subblock(2, 1, 1, 2)
            # for testing
            sig_3 = sig_3 - (np.einsum("Jbc,aJIi->Iiabc", v_ref2, tei_tmp))
            sig_3 = sig_3 + (np.einsum("Jac,bJIi->Iiabc", v_ref2, tei_tmp)) #P(ab)
            sig_3 = sig_3 + (np.einsum("Jba,cJIi->Iiabc", v_ref2, tei_tmp)) #P(ac)
            #sig_3 = sig_3 - (np.einsum("Jca,bJIi->Iiabc", v_ref2, tei_tmp)) #P(ab)P(ac)
            # for testing
            sig_3_2 = -1.0*(np.einsum("Jbc,aJIi->Iiabc", v_ref2, tei_tmp))
            sig_3_2 = sig_3_2 + (np.einsum("Jac,bJIi->Iiabc", v_ref2, tei_tmp)) #P(ab)
            sig_3_2 = sig_3_2 + (np.einsum("Jba,cJIi->Iiabc", v_ref2, tei_tmp)) #P(ac)
            sig_3_2 = sig_3_2 - (np.einsum("Jca,bJIi->Iiabc", v_ref2, tei_tmp)) #P(ab)P(ac)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 
            #   sig(Iiabc:babbb) += -t(Jiabc:babbb)*F(JI:bb)
            Fb_tmp = Fb[0:nb_occ, 0:nb_occ]
            sig_3 = sig_3 - np.einsum("Jiabc,JI->Iiabc", v_ref3, Fb_tmp)
            #   sig(Iiabc:babbb) += -t(Ijabc:babbb)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Ijabc,ji->Iiabc", v_ref3, Fa_tmp)
            #   sig(Iiabc:babbb) += P(ab)*P(ac)*t(Iidbc:babbb)*F(ad:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Iidbc,ad->Iiabc", v_ref3, Fb_tmp)
            sig_3 = sig_3 - np.einsum("Iidac,bd->Iiabc", v_ref3, Fb_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("Iidba,cd->Iiabc", v_ref3, Fb_tmp) #P(ac)
            #sig_3 = sig_3 + np.einsum("Iidca,bd->Iiabc", v_ref3, Fb_tmp) #P(ab)P(ac)
            #   sig(Iiabc:babbb) += P(ab)*P(ac)*t(Jidbc:babbb)*I(aJId:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 1, 1, 2)
            tei_tmp_K = self.tei.get_subblock(2, 1, 2, 1)
            sig_3 = sig_3 + (np.einsum("Jidbc,aJId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jidbc,aJdI->Iiabc", v_ref3, tei_tmp_K))
            sig_3 = sig_3 - (np.einsum("Jidac,bJId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jidac,bJdI->Iiabc", v_ref3, tei_tmp_K)) #P(ab)
            sig_3 = sig_3 - (np.einsum("Jidba,cJId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jidba,cJdI->Iiabc", v_ref3, tei_tmp_K)) #P(ac)
            #sig_3 = sig_3 + (np.einsum("Jidca,bJId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jidca,bJdI->Iiabc", v_ref3, tei_tmp_K)) #P(ab)P(ac)
            #   sig(Iiabc:babbb) += -P(ab)*P(bc)*t(Ijadc:babbb)*I(bjdi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ijadc,bjdi->Iiabc", v_ref3, tei_tmp)
            sig_3 = sig_3 + np.einsum("Ijbdc,ajdi->Iiabc", v_ref3, tei_tmp) #P(ab)
            sig_3 = sig_3 + np.einsum("Ijadb,cjdi->Iiabc", v_ref3, tei_tmp) #P(bc)
            #sig_3 = sig_3 - np.einsum("Ijcdb,ajdi->Iiabc", v_ref3, tei_tmp) #P(ab)P(bc)
            #   sig(Iiabc:babbb) += 0.5*P(ac)*P(bc)*t(Iidec:babbb)*I(abde:bbbb)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 + 0.5*(np.einsum("Iidec,abde->Iiabc", v_ref3, tei_tmp) - np.einsum("Iidec,abed->Iiabc", v_ref3, tei_tmp))
            sig_3 = sig_3 - 0.5*(np.einsum("Iidea,cbde->Iiabc", v_ref3, tei_tmp) - np.einsum("Iidea,cbed->Iiabc", v_ref3, tei_tmp)) #P(ac)
            sig_3 = sig_3 - 0.5*(np.einsum("Iideb,acde->Iiabc", v_ref3, tei_tmp) - np.einsum("Iideb,aced->Iiabc", v_ref3, tei_tmp)) #P(bc)
            #sig_3 = sig_3 + 0.5*(np.einsum("Iidea,bcde->Iiabc", v_ref3, tei_tmp) - np.einsum("Iidea,bced->Iiabc", v_ref3, tei_tmp)) #P(ac)P(bc)
            #   sig(Iiabc:babbb) += -P(ba)*P(bc)*t(Jiadc:babbb)*I(JbId:bbbb)
            tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
            tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
            sig_3 = sig_3 - (np.einsum("Jiadc,JbId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jiadc,JbdI->Iiabc", v_ref3, tei_tmp_K))
            sig_3 = sig_3 + (np.einsum("Jibdc,JaId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jibdc,JadI->Iiabc", v_ref3, tei_tmp_K)) #P(ba)
            sig_3 = sig_3 + (np.einsum("Jiadb,JcId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jiadb,JcdI->Iiabc", v_ref3, tei_tmp_K)) #P(bc)
            #sig_3 = sig_3 - (np.einsum("Jicdb,JaId->Iiabc", v_ref3, tei_tmp_J) - np.einsum("Jicdb,JadI->Iiabc", v_ref3, tei_tmp_K)) #P(ba)P(bc)
            #   sig(Iiabc:babbb) += -P(ab)*P(ac)*t(Ijdbc:babbb)*I(ajdi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Ijdbc,ajdi->Iiabc", v_ref3, tei_tmp)
            sig_3 = sig_3 + np.einsum("Ijdac,bjdi->Iiabc", v_ref3, tei_tmp) #P(ab)
            sig_3 = sig_3 + np.einsum("Ijdba,cjdi->Iiabc", v_ref3, tei_tmp) #P(ac)
            #sig_3 = sig_3 - np.einsum("Ijdca,bjdi->Iiabc", v_ref3, tei_tmp) #P(ab)P(ac)
            #   sig(Iiabc:babbb) += t(Jjabc:babbb)*I(JjIi:baba)
            tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
            sig_3 = sig_3 + (np.einsum("Jjabc,JjIi->Iiabc", v_ref3, tei_tmp))
            #   sig(Iiabc:babbb) += -P(ac)*P(bc)*t(Ijabd:babbb)*I(jcid:abab)
            #tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            #sig_3 = sig_3 + np.einsum("Ijabd,jcid->Iiabc", v_ref3, tei_tmp)
            #sig_3 = sig_3 - np.einsum("Ijcbd,jaid->Iiabc", v_ref3, tei_tmp) #P(ac)
            #sig_3 = sig_3 - np.einsum("Ijacd,jbid->Iiabc", v_ref3, tei_tmp) #P(bc)
            #sig_3 = sig_3 + np.einsum("Ijbcd,jaid->Iiabc", v_ref3, tei_tmp) #P(ac)P(bc)

            e1_2 = np.einsum("iab,iab->", sig_1_2, v_ref1)
            e1_3 = np.einsum("iab,iab->", sig_1_3, v_ref1)

            e2_1 = np.einsum("Iab,Iab->", sig_2_1, v_ref2)
            e2_3 = np.einsum("Iab,Iab->", sig_2_3, v_ref2)

            e3_1 = 0.25*np.einsum("Iiabc,Iiabc->", sig_3_1, v_ref3)
            e3_2 = 0.25*np.einsum("Iiabc,Iiabc->", sig_3_2, v_ref3)

            '''
            print("Are E1 and E2 close?")
            print(np.isclose(e1_2, e2_1))
            print("Are E1 and E3 close?")
            print(np.isclose(e1_3, e3_1))
            print(e1_3, e3_1)
            print("Are E2 and E3 close?")
            print(np.isclose(e2_3, e3_2))
            '''

            sig_1_out = np.zeros((v_b1.shape[0], 1))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        if(abs(sig_1[i, a, b] + sig_1[i, b, a]) > 1e-10):
                            print("ERR: REF 1 NOT ANTISYMMETRIC")
                        sig_1_out[index] = sig_1[i, a, b]
                        index = index + 1
            sig_2_out = np.zeros((v_b2.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for a in range(socc):
                    for b in range(a):
                        if(abs(sig_2[I, a, b] + sig_2[I, b, a]) > 1e-10):
                            print("ERR: REF 2 NOT ANTISYMMETRIC")
                        sig_2_out[index] = sig_2[I, a, b]
                        index = index + 1
            sig_3_out = np.zeros((v_b3.shape[0], 1))
            index = 0
            for I in range(nb_occ):
                for i in range(socc):
                    for a in range(socc):
                        for b in range(a):
                            for c in range(b):
                                if(abs(sig_3[I, i, a, b, c] + sig_3[I, i, b, a, c]) > 1e-10):
                                    print("ERR: REF 3 NOT ANTISYMMETRIC")
                                if(abs(sig_3[I, i, a, b, c] + sig_3[I, i, a, c, b]) > 1e-10):
                                    print("ERR: REF 3 NOT ANTISYMMETRIC")
                                sig_3_out[index] = v_ref3[I, i, a, b, c]
                                index = index + 1

            return np.vstack((sig_1_out, sig_2_out, sig_3_out))

        # do excitation scheme: RAS(p)-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space=="p"):
            """ 
                definitions:
                I      doubly occupied
                i,a    singly occupied
                A      doubly unoccupied

                block1 = v(iab:abb)
                block2 = v(Iab:abb)
                block2 = v(Iiabc:babbb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
            """

            n_b1_dets = int(socc * ((socc-1)*(socc)/2))
            n_b2_dets = int(socc * na_virt * socc)
            v_b1 = v[0:n_b1_dets]
            v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets]
            v_b3 = v[n_b1_dets+n_b2_dets:]

            # v(1) unpack to indexing: (iab:abb)
            v_ref1 = np.zeros((socc,socc,socc))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        v_ref1[i, a, b] = v_b1[index]
                        v_ref1[i, b, a] = -1.0*v_b1[index]
                        index = index + 1
            # v(2) unpack to indexing: (Aia:abb)
            v_ref2 = np.reshape(v_b2, (na_virt, socc, socc))
            # v(3) unpack to indexing: (Aijab:aaabb)
            v_ref3 = np.zeros((na_virt, socc, socc, socc, socc))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        for a in range(socc):
                            for b in range(a):
                                v_ref3[A, i, j, a, b] = v_b3[index]
                                v_ref3[A, j, i, a, b] = -1.0*v_b3[index]
                                v_ref3[A, i, j, b, a] = -1.0*v_b3[index]
                                v_ref3[A, j, i, b, a] = v_b3[index]
                                index = index + 1

            ################################################ 
            # Do the following term: OK
            #       H(1,1) v(1) = sig(1)
            ################################################ 
            #   sig(iab:abb) += P(ab)*t(iab:abb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = (np.einsum("icb,ac->iab", v_ref1, Fb_tmp) - np.einsum("ica,bc->iab", v_ref1, Fb_tmp))
            #   sig(iab:abb) += -t(jab:abb)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_1 = sig_1 - np.einsum("jab,ji->iab", v_ref1, Fa_tmp)
            #   sig(iab:abb) += -P(ab)*t(jac:abb)*I(jbic:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_1 = sig_1 - (np.einsum("jac,jbic->iab", v_ref1, tei_tmp) - np.einsum("jbc,jaic->iab", v_ref1, tei_tmp))
            #   sig(iab:abb) += 0.5*t(icd:abb)*I(abcd:abab)
            sig_1 = sig_1 + 0.5*(np.einsum("icd,abcd->iab", v_ref1, tei_tmp) - np.einsum("icd,abdc->iab", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(1,2) v(2) = sig(1)
            ################################################ 
            #   sig(iab:abb) += t(iBb:abb)*F(aB:bb)
            Fb_tmp = Fb[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Bib,aB->iab", v_ref2, Fb_tmp)
            sig_1 = sig_1 - np.einsum("Bia,bB->iab", v_ref2, Fb_tmp) #P(ab)
            #   sig(iab:abb) += t(iAd:abb)*I(abAd:bbbb)
            tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 + (np.einsum("Aid,abAd->iab", v_ref2, tei_tmp_J) - np.einsum("Aid,abdA->iab", v_ref2, tei_tmp_K))
            #   sig(iab:abb) += -t(jAb:abb)*I(ajAi:baba)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            sig_1 = sig_1 - (np.einsum("Ajb,ajAi->iab", v_ref2, tei_tmp))
            sig_1 = sig_1 + (np.einsum("Aja,bjAi->iab", v_ref2, tei_tmp)) #P(ab)

            ################################################ 
            # Do the following term:
            #       H(1,3) v(3) = sig(1)
            ################################################ 
            # TESTING
            #   sig(iab:abb) += t(jiAab:aabbb)*F(jA:aa)
            Fa_tmp = Fa[nb_occ:na_occ, na_occ:nbf]
            sig_1 = sig_1 + np.einsum("Ajiab,jA->iab", v_ref3, Fa_tmp)
            #   sig(iab:abb) += -1.0*t(jkAab:aabbb)*I(jkAi:aaaa)
            tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
            sig_1 = sig_1 - 0.5*(np.einsum("Ajkab,jkAi->iab", v_ref3, tei_tmp_J) - np.einsum("Ajkab,jkiA->iab", v_ref3, tei_tmp_K))
            #   sig(iab:abb) += t(jiAcb:aabbb)*I(jaAc:aaaa)
            tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
            sig_1 = sig_1 + np.einsum("Ajicb,jaAc->iab", v_ref3, tei_tmp)
            sig_1 = sig_1 - np.einsum("Ajica,jbAc->iab", v_ref3, tei_tmp) #P(ab)

            ################################################ 
            # Do the following term:
            #       H(2,1) v(1) = sig(2)
            ################################################ 
            #   sig(iAa:abb) += t(iba:abb)*F(Ab:bb)
            Fb_tmp = Fb[na_occ:nbf, nb_occ:na_occ]
            sig_2 = np.einsum("iba,Ab->Aia", v_ref1, Fb_tmp)
            #   sig(iAa:abb) += 0.5*t(ibc:abb)*I(Aabc:bbbb)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_2 = sig_2 + 0.5*(np.einsum("ibc,Aabc->Aia", v_ref1, tei_tmp) - np.einsum("ibc,Aacb->Aia", v_ref1, tei_tmp))
            #   sig(iab:abb) += -t(jab:abb)*I(Ajai:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_2 = sig_2 - (np.einsum("jba,Ajbi->Aia", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(2,2) v(2) = sig(2)
            ################################################ 
            #   sig(iAa:abb) += t(iBa:abb)*F(AB:bb)
            Fb_tmp = Fb[na_occ:nbf, na_occ:nbf]
            sig_2 = sig_2 + np.einsum("Bia,AB->Aia", v_ref2, Fb_tmp)
            #   sig(iAa:abb) += -t(jAa:abb)*F(ji:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 - np.einsum("Aja,ji->Aia", v_ref2, Fa_tmp)
            #   sig(iAa:abb) += -t(jAa:abb)*F(ji:aa)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_2 = sig_2 + np.einsum("Aib,ab->Aia", v_ref2, Fb_tmp)
            #   sig(iAa:abb) += t(iBb:abb)*I(AaBb:bbbb)
            tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
            sig_2 = sig_2 + (np.einsum("Bib,AaBb->Aia", v_ref2, tei_tmp_J) - np.einsum("Bib,AabB->Aia", v_ref2, tei_tmp_K))
            #   sig(iAa:abb) += -1.0*t(jAb:abb)*I(jaib:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_2 = sig_2 - np.einsum("Ajb,jaib->Aia", v_ref2, tei_tmp)
            #   sig(iAa:abb) += -1.0*t(jBb:abb)*I(AjBi:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_2 = sig_2 - np.einsum("Bja,AjBi->Aia", v_ref2, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(2,3) v(3) = sig(2)
            ################################################ 
            #   sig(iAb:abb) += t(jiBba:aabbb)*I(jABb:aaaa)
            tei_tmp = self.tei.get_subblock(2, 3, 3, 2)
            sig_2 = sig_2 + np.einsum("Bjiba,jABb->Aia", v_ref3, tei_tmp)

            ################################################ 
            # Do the following term:
            #       H(3,1) v(1) = sig(3)
            ################################################ 
            #   sig(ijAab:aaabb) += t(jab:abb)*F(Ai:aa)
            Fa_tmp = Fa[na_occ:nbf, nb_occ:na_occ]
            sig_3 = np.einsum("jab,Ai->Aijab", v_ref1, Fa_tmp)
            sig_3 = sig_3 - np.einsum("iab,Aj->Aijab", v_ref1, Fa_tmp) #P(ij)
            #   sig(ijAab:aaabb) += P(ab)*t(jcb:abb)*I(Aaic:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
            sig_3 = sig_3 + np.einsum("jcb,Aaic->Aijab", v_ref1, tei_tmp)
            sig_3 = sig_3 - np.einsum("jca,Abic->Aijab", v_ref1, tei_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("icb,Aajc->Aijab", v_ref1, tei_tmp) #P(ij)
            sig_3 = sig_3 + np.einsum("ica,Abjc->Aijab", v_ref1, tei_tmp) #P(ab)P(ij)
            #   sig(ijAab:aaabb) += -1.0*t(kab:abb)*I(Akij:aaaa)
            sig_3 = sig_3 - (np.einsum("kab,Akij->Aijab", v_ref1, tei_tmp) - np.einsum("kab,Akji->Aijab", v_ref1, tei_tmp))

            ################################################ 
            # Do the following term:
            #       H(3,2) v(2) = sig(3)
            ################################################ 
            #   sig(ijAab:aaabb) += P(ab)*t(jBb:abb)*I(AaiB:abab)
            tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 + np.einsum("Bjb,AaiB->Aijab", v_ref2, tei_tmp)
            sig_3 = sig_3 - np.einsum("Bja,AbiB->Aijab", v_ref2, tei_tmp) #P(ab)
            sig_3 = sig_3 - np.einsum("Bib,AajB->Aijab", v_ref2, tei_tmp) #P(ij)
            sig_3 = sig_3 + np.einsum("Bia,AbjB->Aijab", v_ref2, tei_tmp) #P(ab)P(ij)

            ################################################ 
            # Do the following term:
            #       H(3,3) v(3) = sig(3)
            ################################################ 
            #   sig(ijAab:aaabb) += -1.0*P(ij)*t(kjAab:aabbb)*F(ki:aa)
            Fa_tmp = Fa[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 - np.einsum("Akjab,ki->Aijab", v_ref3, Fa_tmp)
            sig_3 = sig_3 + np.einsum("Akiab,kj->Aijab", v_ref3, Fa_tmp) #P(ij)
            #   sig(ijAab:aaabb) += t(ijBab:aabbb)*F(AB:aa)
            Fa_tmp = Fa[na_occ:nbf, na_occ:nbf]
            sig_3 = sig_3 + np.einsum("Bijab,AB->Aijab", v_ref3, Fa_tmp)
            #   sig(ijAab:aaabb) += P(ab)*t(ijAcb:aabbb)*F(ac:bb)
            Fb_tmp = Fb[nb_occ:na_occ, nb_occ:na_occ]
            sig_3 = sig_3 + np.einsum("Aijcb,ac->Aijab", v_ref3, Fb_tmp)
            sig_3 = sig_3 - np.einsum("Aijca,bc->Aijab", v_ref3, Fb_tmp) #P(ab)
            #   sig(ijAab:aaabb) += P(ab)*t(ijBcb:aabbb)*I(AaBc:baba)
            tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
            sig_3 = sig_3 + np.einsum("Bijcb,AaBc->Aijab", v_ref3, tei_tmp) 
            sig_3 = sig_3 - np.einsum("Bijca,AbBc->Aijab", v_ref3, tei_tmp) #P(ab)
            #   sig(ijAab:aaabb) += -1.0*P(ij)*t(ikBab:aabbb)*I(AkBj:aaaa)
            tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
            tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
            sig_3 = sig_3 - (np.einsum("Bikab,AkBj->Aijab", v_ref3, tei_tmp_J) - np.einsum("Bikab,AkjB->Aijab", v_ref3, tei_tmp_K))
            sig_3 = sig_3 + (np.einsum("Bjkab,AkBi->Aijab", v_ref3, tei_tmp_J) - np.einsum("Bjkab,AkiB->Aijab", v_ref3, tei_tmp_K))
            #   sig(ijAab:aaabb) += -1.0*P(ij)*P(ab)*t(kjAcb:aabbb)*I(kaic:abab)
            tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
            sig_3 = sig_3 - np.einsum("Akjcb,kaic->Aijab", v_ref3, tei_tmp) 
            sig_3 = sig_3 + np.einsum("Akjca,kbic->Aijab", v_ref3, tei_tmp) #P(ab)
            sig_3 = sig_3 + np.einsum("Akicb,kajc->Aijab", v_ref3, tei_tmp) #P(ij)
            sig_3 = sig_3 - np.einsum("Akica,kbjc->Aijab", v_ref3, tei_tmp) #P(ab)P(ij)
            #   sig(ijAab:aaabb) += 0.5*t(klAab:aabbb)*I(klij:aaaa)
            sig_3 = sig_3 + 0.5*(np.einsum("Aklab,klij->Aijab", v_ref3, tei_tmp) - np.einsum("Aklab,klji->Aijab", v_ref3, tei_tmp))
            #   sig(ijAab:aaabb) += 0.5*t(ijAcd:aabbb)*I(abcd:bbbb)
            sig_3 = sig_3 + 0.5*(np.einsum("Aijcd,abcd->Aijab", v_ref3, tei_tmp) - np.einsum("Aijcd,abdc->Aijab", v_ref3, tei_tmp))

            # v(1) unpack to indexing: (iab:abb)
            sig_1_out = np.zeros((v_b1.shape[0], 1))
            index = 0
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        sig_1_out[index] = sig_1[i, a, b]
                        index = index + 1
            # v(2) unpack to indexing: (iAa:abb)
            sig_2_out = np.reshape(sig_2, (v_b2.shape[0], 1))
            # v(3) unpack to indexing: (ijAab:aaabb)
            sig_3_out = np.zeros((v_b3.shape[0], 1))
            index = 0
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        for a in range(socc):
                            for b in range(a):
                                sig_3_out[index] = sig_3[A, i, j, a, b]
                                index = index + 1

            return np.vstack((sig_1_out, sig_2_out, sig_3_out))

    # These two are vestigial-- I'm sure they served some purpose in the parent class,
    # but we only really need matvec for our purposes!
    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)
    def _matmat(self, v):
        print("matvec function called -- not implemented yet!!")
        return np.zeros(30)


