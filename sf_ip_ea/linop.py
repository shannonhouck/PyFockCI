import numpy as np
from scipy.sparse.linalg import LinearOperator
from .post_ci_analysis import generate_dets

class LinOpH (LinearOperator):
    
    def __init__(self, shape_in, offset_in, ras1_in, ras2_in, ras3_in,
                 Fa_in, Fb_in, tei_in, n_SF_in, delta_ec_in, conf_space_in):
        super(LinOpH, self).__init__(dtype=np.dtype('float64'), shape=shape_in)
        # getting the numbers of orbitals
        self.offset = offset_in # diagonal offset (usually energy)
        self.ras1 = ras1_in # number of beta occupied
        self.ras2 = ras2_in # number of beta occupied
        self.ras3 = ras3_in # number of alpha virtual
        # setting useful parameters
        self.n_SF = n_SF_in # number of spin-flips
        self.delta_ec = delta_ec_in # change in electron count
        self.conf_space = conf_space_in # excitation rank
        self.num_dets = shape_in[0]
        # getting integrals
        self.Fa = Fa_in
        self.Fb = Fb_in
        self.tei = tei_in

    def diag(self):
        """Returns approximate diagonal of Hamiltonian for Davidson.
        """
        # grabbing necessary info from self
        n_dets = self.num_dets
        Fa = self.Fa
        Fb = self.Fb
        tei = self.tei
        conf_space = self.conf_space
        offset = self.offset
        ras1 = self.ras1
        ras2 = self.ras2
        ras3 = self.ras3
        n_SF = self.n_SF
        delta_ec = self.delta_ec
        nbf = ras1 + ras2 + ras3
        # get list of determinants
        det_list = generate_dets(n_SF, delta_ec, conf_space, ras1,
                                 ras2, ras3) 
        # building "base" value
        Fa_tmp = Fa[0:ras1+ras2, 0:ras1+ras2]
        Fb_tmp = Fb[0:ras1, 0:ras1]
        base = offset
        # set up diagonal
        diag_out = np.zeros((n_dets))
        # replace necessary values
        count = 0
        for det in det_list:
            diag_out[count] = base
            # eliminate electrons
            for i in det[0][0]:
                diag_out[count] = diag_out[count] - Fa[i,i]
            for i in det[0][1]:
                diag_out[count] = diag_out[count] - Fb[i,i]
            # add electrons
            for a in det[1][0]:
                diag_out[count] = diag_out[count] + Fa[a,a]
            for a in det[1][1]:
                diag_out[count] = diag_out[count] + Fb[a,a]
            count = count + 1
        return diag_out

    def do_cas_1sf(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do CAS-1SF.

           Evaluate the following matrix vector multiply:
                | H(1,1) | * v(1) = sig(1)
           
        """
        ################################################ 
        # Put guess vector into block form
        ################################################ 
        v_b1 = np.reshape(v, (ras2,ras2,v.shape[1])) # v for block 1

        ################################################ 
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ia:ba) += -v(ia:ba) (eps(a:b)-eps(i:a))
        Fi_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fa_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("ibn,ba->ian", v_b1, Fa_tmp) 
        sig_1 = sig_1 - np.einsum("jan,ji->ian", v_b1, Fi_tmp)
        #   sig(ai:ba) += -v(bj:ba) I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("jbn,ajbi->ian", v_b1, tei_tmp)
        # using reshape because non-contiguous in memory
        # look into this while doing speedup
        sig_1 = np.reshape(sig_1, (v.shape[0], v.shape[1]))
        return sig_1 + offset_v

    def do_cas_1sf_neutral(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do CAS-1SF.

           Evaluate the following matrix vector multiply:
                | H(1,1) | * v(1) = sig(1)
           
        """
        ################################################ 
        # Put guess vector into block form
        ################################################ 
        v_b1 = v # v for block 1

        ################################################ 
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ia:ba) += -v(ia:ba) (eps(a:b)-eps(i:a))
        Fi_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fa_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("in,ii->in", v_b1, Fa_tmp)
        sig_1 = sig_1 - np.einsum("in,ii->in", v_b1, Fi_tmp)
        #   sig(ai:ba) += -v(bj:ba) I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("jn,ijji->in", v_b1, tei_tmp)
        # using reshape because non-contiguous in memory
        # look into this while doing speedup
        #sig_1 = np.reshape(sig_1, (v.shape[0], v.shape[1]))
        return sig_1 + offset_v

    def do_cas_2sf(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do CAS-2SF.

                block1 = v(ijab:aabb)

                Evaluate the following matrix vector multiply:

                | H(1,1) | * v(1) = sig(1)
           
        """
        # v(1) unpack to indexing: (ijab:aabb)
        v_ref1 = np.zeros((ras2, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        for b in range(a):
                            v_ref1[i, j, a, b, n] = v[index, n]
                            v_ref1[i, j, b, a, n] = -1.0*v[index, n]
                            v_ref1[j, i, a, b, n] = -1.0*v[index, n]
                            v_ref1[j, i, b, a, n] = v[index, n]
                            index = index + 1

        ################################################ 
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ijab:aabb) += -v(kjab:aabb) F(ik:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*np.einsum("kjabn,ki->ijabn", v_ref1, Fa_tmp)
        sig_1 = sig_1 + np.einsum("kiabn,kj->ijabn", v_ref1, Fa_tmp) #P(ij)
        #   sig(ijab:aabb) += v(ijcb:aabb) F(ac:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("ijcbn,ac->ijabn", v_ref1, Fb_tmp)
        sig_1 = sig_1 - np.einsum("ijcan,bc->ijabn", v_ref1, Fb_tmp) #P(ab)
        #   sig(ijab:aabb) += 0.5*v(ijcd:aabb) I(abcd:bbbb)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 + 0.5*(np.einsum("ijcdn,abcd->ijabn", v_ref1, tei_tmp)
                             - np.einsum("ijcdn,abdc->ijabn", v_ref1, tei_tmp))
        #   sig(ijab:aabb) += 0.5*v(klab:aabb) I(klij:aaaa)
        sig_1 = sig_1 + 0.5*(np.einsum("klabn,klij->ijabn", v_ref1, tei_tmp)
                             - np.einsum("klabn,klji->ijabn", v_ref1, tei_tmp))
        #   sig(ijab:aabb) += - P(ij)P(ab) v(ikcb:aabb) I(akcj:baba)
        sig_1 = sig_1 - (np.einsum("ikcbn,akcj->ijabn", v_ref1, tei_tmp))
        sig_1 = sig_1 + (np.einsum("jkcbn,akci->ijabn", v_ref1, tei_tmp)) #P(ij)
        sig_1 = sig_1 + (np.einsum("ikcan,bkcj->ijabn", v_ref1, tei_tmp)) #P(ab)
        sig_1 = sig_1 - (np.einsum("jkcan,bkci->ijabn", v_ref1, tei_tmp)) #P(ij)P(ab)

        sig_1_out = np.zeros((v.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        for b in range(a):
                            sig_1_out[index, n] = sig_1[i, j, a, b, n]
                            index = index + 1

        return sig_1_out + offset_v

    def do_h_1sf(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do RAS(h)-1SF.

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
        v_b1 = v[:(ras2*ras2), :] # v for block 1
        v_b2 = v[(ras2*ras2):((ras2*ras2)+(ras2*ras1)), :] # v for block 2
        v_b3 = v[((ras2*ras2)+(ras2*ras1)):, :] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (ras2, ras2, v.shape[1]))
        # v(2) indexing: (Ia:ab)
        v_ref2 = np.reshape(v_b2, (ras1, ras2, v.shape[1]))
        # v(3) unpack to indexing: (Iiab:babb)
        v_ref3 = np.zeros((ras1, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for a in range(ras2):
                        for b in range(a):
                            v_ref3[I, i, a, b, n] = v_b3[index, n]
                            v_ref3[I, i, b, a, n] = -1.0*v_b3[index, n]
                            index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 

        #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("ibn,ab->ian", v_ref1, Fb_tmp) - np.einsum("jan,ji->ian", v_ref1, Fa_tmp)
        #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("jbn,ajbi->ian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 

        #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
        Fa_tmp = Fa[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("Ian,Ii->ian", v_ref2, Fa_tmp)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ibn,Iaib->ian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 

        #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, 0:ras1]
        sig_1 = sig_1 + np.einsum("Iiban,bI->ian", v_ref3, Fb_tmp)
        #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ijban,Ijbi->ian", v_ref3, tei_tmp)
        #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 + 0.5*(np.einsum("Iicbn,Iacb->ian", v_ref3, tei_tmp)
                             - np.einsum("Iicbn,Iabc->ian", v_ref3, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, 0:ras1]
        sig_2 = -1.0*np.einsum("ian,iI->Ian", v_ref1, Fa_tmp)
        #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
        sig_2 = sig_2 - np.einsum("ibn,iaIb->Ian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += sig(Ia:ab)*F(ab:bb) - sig(Ja:ab)*F(IJ:aa)
        Fa_tmp = Fa[0:ras1, 0:ras1]
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + np.einsum("Ibn,ab->Ian", v_ref2, Fb_tmp) - np.einsum("Jan,IJ->Ian", v_ref2, Fa_tmp)
        #   sig(Ia:ab) += v(Jb:ab)*I(JaIb:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_2 = sig_2 - np.einsum("Jbn,JaIb->Ian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += -v(Jiab:babb)*I(JibI:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 + np.einsum("Jiabn,JibI->Ian", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 

        #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
        Fb_tmp = Fb[0:ras1, ras1:ras1+ras2]
        sig_3 = (np.einsum("ibn,Ia->Iiabn", v_ref1, Fb_tmp) - np.einsum("ian,Ib->Iiabn", v_ref1, Fb_tmp))
        #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
        sig_3 = sig_3 + np.einsum("jan,jbiI->Iiabn", v_ref1, tei_tmp) - np.einsum("jbn,jaiI->Iiabn", v_ref1, tei_tmp)
        #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
        sig_3 = sig_3 + np.einsum("icn,abIc->Iiabn", v_ref1, tei_tmp_J) - np.einsum("icn,abcI->Iiabn", v_ref1, tei_tmp_K)

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 

        #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_3 = sig_3 + np.einsum("Jan,JbiI->Iiabn", v_ref2, tei_tmp)
        sig_3 = sig_3 - np.einsum("Jbn,JaiI->Iiabn", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 

        #   sig(Iiab:babb) += t(Iiac:babb)*F(bc:bb) - t(Iibc:babb)*F(ac:bb)
        F_ac_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 + np.einsum("Iiacn,bc->Iiabn", v_ref3, F_ac_tmp)
        sig_3 = sig_3 - np.einsum("Iibcn,ac->Iiabn", v_ref3, F_ac_tmp)
        #   sig(Iiab:babb) += -1.0*t(Ijab:babb)*F(ij:aa)
        F_ij_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Ijabn,ij->Iiabn", v_ref3, F_ij_tmp)
        #   sig(Iiab:babb) += -1.0*t(Jiab:babb)*F(IJ:bb)
        F_IJ_tmp = Fb[0:ras1, 0:ras1]
        sig_3 = sig_3 - np.einsum("Jiabn,IJ->Iiabn", v_ref3, F_IJ_tmp)
        #   sig(Iiab:babb) += 0.5*v(Iicd:babb)*I(abcd:bbbb)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 + 0.5*(np.einsum("Iicdn,abcd->Iiabn", v_ref3, tei_tmp)
                             - np.einsum("Iicdn,abdc->Iiabn", v_ref3, tei_tmp))
        #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("Ijcbn,ajci->Iiabn", v_ref3, tei_tmp)
        sig_3 = sig_3 + np.einsum("Ijcan,bjci->Iiabn", v_ref3, tei_tmp)
        #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_3 = sig_3 - (np.einsum("Jiacn,JbIc->Iiabn", v_ref3, tei_tmp_J)
                         - np.einsum("Jiacn,JbcI->Iiabn", v_ref3, tei_tmp_K))
        #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_3 = sig_3 + (np.einsum("Jibcn,JaIc->Iiabn", v_ref3, tei_tmp_J)
                         - np.einsum("Jibcn,JacI->Iiabn", v_ref3, tei_tmp_K))
        #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_3 = sig_3 + np.einsum("Jjabn,JjIi->Iiabn", v_ref3, tei_tmp)

        ################################################ 
        # sigs complete-- free to reshape!
        ################################################ 
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        # pack sig(3) vector for returning
        sig_3_out = np.zeros((v_b3.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for a in range(ras2):
                        for b in range(a):
                            sig_3_out[index, n] = sig_3[I, i, a, b, n]
                            index = index + 1

        # combine and return
        return np.vstack((sig_1, sig_2, sig_3_out)) + offset_v

    def do_p_1sf(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do RAS(p)-1SF.

                block1 = v(ai:ba)
                block2 = v(Ai:ba)
                block3 = v(Aaij:abaa)

                Evaluate the following matrix vector multiply:

                H(1,1) * v(1) + H(1,2) * v(2) + H(1,3) * v(3) = sig(1)
                H(2,1) * v(1) + H(2,2) * v(2) + H(2,3) * v(3) = sig(2)
                H(3,1) * v(1) + H(3,2) * v(2) + H(3,3) * v(3) = sig(3)
           
        """
        nbf = ras1 + ras2 + ras3
        ################################################ 
        # Separate guess vector into blocks 1, 2, and 3
        ################################################ 
        v_b1 = v[:(ras2*ras2), :] # v for block 1
        v_b2 = v[(ras2*ras2):(ras2*(ras2+ras3)), :] # v for block 2
        v_b3 = v[(ras2*(ras2+ras3)):, :] # v for block 3
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (ras2, ras2, v.shape[1]))
        # v(2) indexing: (Ai:ab)
        v_ref2 = np.reshape(v_b2, (ras3, ras2, v.shape[1]))
        # v(3) unpack to indexing: (Aijb:aaab)
        v_ref3 = np.zeros((ras3, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for b in range(ras2):
                            v_ref3[A, i, j, b, n] = v_b3[index, n]
                            v_ref3[A, j, i, b, n] = -1.0*v_b3[index, n]
                            index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
       
        #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("ibn,ab->ian", v_ref1, Fb_tmp) - np.einsum("jan,ji->ian", v_ref1, Fa_tmp)
        #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("jbn,ajbi->ian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 

        #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Ain,aA->ian", v_ref2, Fb_tmp)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        sig_1 = sig_1 - np.einsum("Bjn,ajBi->ian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 

        #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Ajian,jA->ian", v_ref3, Fa_tmp)
        #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - np.einsum("Aijbn,ajbA->ian", v_ref3, tei_tmp)
        #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - 0.5*(np.einsum("Ajkan,jkAi->ian", v_ref3, tei_tmp)
                             - np.einsum("Ajkan,jkiA->ian", v_ref3, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 

        #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
        Fb_tmp = Fb[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_2 = np.einsum("ibn,Ab->Ain", v_ref1, Fb_tmp)
        #   sig(iA:ab) += v(jb:ab)*t(jb:ab)*I(Ajbi:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_2 = sig_2 - np.einsum("jbn,Ajbi->Ain", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 

        #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fb_tmp = Fb[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_2 = sig_2 + np.einsum("Bin,AB->Ain", v_ref2, Fb_tmp) - np.einsum("Ajn,ji->Ain", v_ref2, Fa_tmp)
        #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_2 = sig_2 - np.einsum("Bjn,AjBi->Ain", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 

        #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_2 = sig_2 - np.einsum("Bijcn,AjcB->Ain", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 

        #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_3 = (np.einsum("jbn,iA->Aijbn", v_ref1, Fa_tmp) - np.einsum("ibn,jA->Aijbn", v_ref1, Fa_tmp))
        #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("icn,Abjc->Aijbn", v_ref1, tei_tmp) + np.einsum("jcn,Abic->Aijbn", v_ref1, tei_tmp)
        #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_3 = sig_3 - (np.einsum("kbn,Akij->Aijbn", v_ref1, tei_tmp) - np.einsum("kbn,Akji->Aijbn", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 

        #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 - np.einsum("Cin,AbjC->Aijbn", v_ref2, tei_tmp)
        sig_3 = sig_3 + np.einsum("Cjn,AbiC->Aijbn", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 

        #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
        F_bc_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        F_AB_tmp = Fa[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_3 = sig_3 + np.einsum("Aijcn,bc->Aijbn", v_ref3, F_bc_tmp) # no contribution
        sig_3 = sig_3 + np.einsum("Bijbn,AB->Aijbn", v_ref3, F_AB_tmp) # no contribution
        Fi_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Aikbn,kj->Aijbn", v_ref3, Fi_tmp) + np.einsum("Ajkbn,ki->Aijbn", v_ref3, Fi_tmp)
        #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_3 = sig_3 + np.einsum("Bijcn,AbBc->Aijbn", v_ref3, tei_tmp)
        #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 + 0.5*(np.einsum("Aklbn,klij->Aijbn", v_ref3, tei_tmp) - np.einsum("Aklbn,klji->Aijbn", v_ref3, tei_tmp))
        #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
        sig_3 = sig_3 - np.einsum("Akjcn,kbic->Aijbn", v_ref3, tei_tmp) 
        #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
        sig_3 = sig_3 + np.einsum("Akicn,kbjc->Aijbn", v_ref3, tei_tmp)
        #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
        tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 - (np.einsum("Cikbn,AkCj->Aijbn", v_ref3, tei_tmp_J) - np.einsum("Cikbn,AkjC->Aijbn", v_ref3, tei_tmp_K))
        #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
        sig_3 = sig_3 + (np.einsum("Cjkbn,AkCi->Aijbn", v_ref3, tei_tmp_J) - np.einsum("Cjkbn,AkiC->Aijbn", v_ref3, tei_tmp_K))

        ################################################ 
        # sigs complete-- free to reshape!
        ################################################ 
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        # pack sig(3) vector for returning
        sig_3_out = np.zeros((v_b3.shape[0], v.shape[1])) # add 0.5
        for n in range(v.shape[1]):
            index = 0 
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for b in range(ras2):
                            sig_3_out[index] = sig_3[A, i, j, b]  
                            index = index + 1 

        # combine and return
        return np.vstack((sig_1, sig_2, sig_3_out)) + offset_v

    def do_hp_1sf(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do RAS(h,p)-1SF.

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
        nbf = ras1 + ras2 + ras3
        ################################################ 
        # Separate guess vector into blocks 1, 2, and 3
        ################################################ 
        n_b1_dets = int(ras2*ras2)
        n_b2_dets = int(ras1*ras2)
        n_b3_dets = int(ras2*ras3)
        n_b4_dets = int(ras1*ras2*(ras2*(ras2-1)/2))
        n_b5_dets = int(ras3*ras2*(ras2*(ras2-1)/2))
        v_b1 = v[0:n_b1_dets, :] # v for block 1
        v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets, :] # v for block 2
        v_b3 = v[n_b1_dets+n_b2_dets:n_b1_dets+n_b2_dets+n_b3_dets, :] # v for block 3
        v_b4 = v[n_b1_dets+n_b2_dets+n_b3_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets, :] # v for block 4
        v_b5 = v[n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets:n_b1_dets+n_b2_dets+n_b3_dets+n_b4_dets+n_b5_dets, :] # v for block 5
        # v(1) indexing: (ia:ab)
        v_ref1 = np.reshape(v_b1, (ras2, ras2, v.shape[1]))
        # v(2) indexing: (Ia:ab)
        v_ref2 = np.reshape(v_b2, (ras1, ras2, v.shape[1]))
        # v(3) indexing: (Ai:ab)
        v_ref3 = np.reshape(v_b3, (ras3, ras2, v.shape[1]))
        # v(4) unpack to indexing: (Iiab:babb)
        v_ref4 = np.zeros((ras1, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for a in range(ras2):
                        for b in range(a):
                            v_ref4[I, i, a, b, n] = v_b4[index, n]
                            v_ref4[I, i, b, a, n] = -1.0*v_b4[index, n]
                            index = index + 1
        # v(5) unpack to indexing: (Aijb:aaab)
        v_ref5 = np.zeros((ras3, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for b in range(ras2):
                            v_ref5[A, i, j, b, n] = v_b5[index, n]
                            v_ref5[A, j, i, b, n] = -1.0*v_b5[index, n]
                            index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 

        #   sig(ia:ab) += v(ib:ab)*F(ab:bb) - v(ja:ab)*F(ij:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("ibn,ab->ian", v_ref1, Fb_tmp) - np.einsum("jan,ji->ian", v_ref1, Fa_tmp)
        #   sig(ia:ab) += v(jb:ab)*I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("jbn,ajbi->ian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 

        #   sig(ia:ab) += sig(Ia:ab)*F(iI:aa)
        Fa_tmp = Fa[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("Ian,Ii->ian", v_ref2, Fa_tmp)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(Iaib:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ibn,Iaib->ian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 
        
        #   sig(ia:ab) += sig(iA:ab)*F(aA:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Ain,aA->ian", v_ref3, Fb_tmp)
        #   sig(ia:ab) += -1.0*sig(jB:ab)*I(ajBi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        sig_1 = sig_1 - np.einsum("Bjn,ajBi->ian", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,4) v(4) = sig(1)
        ################################################ 

        #   sig(ia:ab) += v(Iiba:babb)*F(Ib:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, 0:ras1]
        sig_1 = sig_1 + np.einsum("Iiban,bI->ian", v_ref4, Fb_tmp)
        #   sig(iA:ab) += -1.0*v(Ijba:babb)*I(Ijbi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ijban,Ijbi->ian", v_ref4, tei_tmp)
        #   sig(iA:ab) += 0.5*v(Iicb:babb)*I(Iacb:bbbb)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 + 0.5*(np.einsum("Iicbn,Iacb->ian", v_ref4, tei_tmp)
                             - np.einsum("Iicbn,Iabc->ian", v_ref4, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(1,5) v(5) = sig(1)
        ################################################ 

        #   sig(iA:ab) += v(ijAb:aaab)*F(jA:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Ajian,jA->ian", v_ref5, Fa_tmp)
        #   sig(iA:ab) += - v(ijAb:aaab)*I(ajbA:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - np.einsum("Aijbn,ajbA->ian", v_ref5, tei_tmp)
        #   sig(iA:ab) += -0.5*v(jkAb:aaab)*I(jkia:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - 0.5*(np.einsum("Ajkan,jkAi->ian", v_ref5, tei_tmp)
                             - np.einsum("Ajkan,jkiA->ian", v_ref5, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += -1.0*v(ia:ab)*F(iI:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, 0:ras1]
        sig_2 = -1.0*np.einsum("ian,iI->Ian", v_ref1, Fa_tmp)
        #   sig(iA:ab) += -1.0*v(jb:ab)*t(ib:ab)*I(iaIb:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
        sig_2 = sig_2 - np.einsum("ibn,iaIb->Ian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += sig(Ia:ab)*F(ab:bb) - sig(Ja:ab)*F(IJ:aa)
        Fa_tmp = Fa[0:ras1, 0:ras1]
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + np.einsum("Ibn,ab->Ian", v_ref2, Fb_tmp) - np.einsum("Jan,IJ->Ian", v_ref2, Fa_tmp)
        #   sig(Ia:ab) += v(Jb:ab)*I(JaIb:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_2 = sig_2 - np.einsum("Jbn,JaIb->Ian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += -1.0*v(iA:ab)*I(aiAI:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 1)
        sig_2 = sig_2 - np.einsum("Ain,aiAI->Ian", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,4) v(4) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += - v(Jiab:babb)*I(JibI:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 + np.einsum("Jiabn,JibI->Ian", v_ref4, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,5) v(5) = sig(2)
        ################################################ 

        #   sig(Ia:ab) += - v(ijAa:aaab)*I(ijAI:aaaa)
        tei_tmp_J = self.tei.get_subblock(2, 2, 3, 1)
        tei_tmp_K = self.tei.get_subblock(2, 2, 1, 3)
        sig_2 = sig_2 - 0.5*(np.einsum("Aijan,ijAI->Ian", v_ref5, tei_tmp_J) - np.einsum("Aijan,ijIA->Ian", v_ref5, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 

        #   sig(iA:ab) += v(ib:ab)*F(Ab:bb)
        Fb_tmp = Fb[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_3 = np.einsum("ibn,Ab->Ain", v_ref1, Fb_tmp)
        #   sig(iA:ab) += v(jb:ab)*I(Ajbi:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("jbn,Ajbi->Ain", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 

        #   sig(iA:ab) += -v(Ia:ab)*I(AIai:baba)
        tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
        sig_3 = sig_3 - np.einsum("Ian,AIai->Ain", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 

        #   sig(iA:ab) += sig(iB:ab)*F(BA:bb) - sig(jA:ab)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        Fb_tmp = Fb[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_3 = sig_3 + np.einsum("Bin,AB->Ain", v_ref3, Fb_tmp) - np.einsum("Ajn,ji->Ain", v_ref3, Fa_tmp)
        #   sig(iA:ab) += v(jB:ab)*I(AjBi:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_3 = sig_3 - np.einsum("Bjn,AjBi->Ain", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,4) v(4) = sig(3)
        ################################################ 

        #   sig(iA:ab) += v(Iibc:babb)*I(IAbc:bbbb)
        tei_tmp = self.tei.get_subblock(1, 3, 2, 2)
        sig_3 = sig_3 + 0.5*(np.einsum("Iibcn,IAbc->Ain", v_ref4, tei_tmp) - np.einsum("Iibcn,IAcb->Ain", v_ref4, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(3,5) v(5) = sig(3)
        ################################################ 

        #   sig(iA:ab) += - v(ijBc:aaab)*I(AjcB:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 - np.einsum("Bijcn,AjcB->Ain", v_ref5, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(4,1) v(1) = sig(4)
        ################################################ 

        #   sig(Iiab:babb) += v(ib:ab)*F(Ia:bb) - v(ia:ab)*F(Ib:bb)
        Fb_tmp = Fb[0:ras1, ras1:ras1+ras2]
        sig_4 = (np.einsum("ibn,Ia->Iiabn", v_ref1, Fb_tmp) - np.einsum("ian,Ib->Iiabn", v_ref1, Fb_tmp))
        #   sig(Iiab:babb) += v(ja:ab)*I(jbiI:abab) - v(jb:ab)*I(jaiI:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
        sig_4 = sig_4 + np.einsum("jan,jbiI->Iiabn", v_ref1, tei_tmp) - np.einsum("jbn,jaiI->Iiabn", v_ref1, tei_tmp)
        #   sig(Iiab:aaab) += v(ic:ab)*I(abIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
        sig_4 = sig_4 + np.einsum("icn,abIc->Iiabn", v_ref1, tei_tmp_J) - np.einsum("icn,abcI->Iiabn", v_ref1, tei_tmp_K)

        ################################################ 
        # Do the following term:
        #       H(4,2) v(2) = sig(4)
        ################################################ 

        #   sig(Iiab:babb) += v(Ja:ab)*I(JbiI:abab) - v(Jb:ab)*I(JaiI:abab)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_4 = sig_4 + np.einsum("Jan,JbiI->Iiabn", v_ref2, tei_tmp)
        sig_4 = sig_4 - np.einsum("Jbn,JaiI->Iiabn", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(4,3) v(3) = sig(4)
        ################################################ 

        #   sig(Iiab:babb) += v(iA:ab)*I(abIA:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 3)
        tei_tmp_K = self.tei.get_subblock(2, 2, 3, 1)
        sig_4 = sig_4 + (np.einsum("Ain,abIA->Iiabn", v_ref3, tei_tmp_J) - np.einsum("Ain,abAI->Iiabn", v_ref3, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(4,4) v(4) = sig(4)
        ################################################ 

        #   sig(Iiab:babb) += t(Iiac:babb)*F(bc:bb) - t(Iibc:babb)*F(ac:bb)
        F_ac_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_4 = sig_4 + np.einsum("Iiacn,bc->Iiabn", v_ref4, F_ac_tmp) - np.einsum("Iibcn,ac->Iiabn", v_ref4, F_ac_tmp)
        #   sig(Iiab:babb) += -1.0*t(Ijab:babb)*F(ij:aa)
        F_ij_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_4 = sig_4 - np.einsum("Ijabn,ij->Iiabn", v_ref4, F_ij_tmp)
        #   sig(Iiab:babb) += -1.0*t(Jiab:babb)*F(IJ:bb)
        F_IJ_tmp = Fb[0:ras1, 0:ras1]
        sig_4 = sig_4 - np.einsum("Jiabn,IJ->Iiabn", v_ref4, F_IJ_tmp)
        #   sig(Iiab:babb) += 0.5*v(Iicd:babb)*I(abcd:bbbb)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_4 = sig_4 + 0.5*(np.einsum("Iicdn,abcd->Iiabn", v_ref4, tei_tmp)
                             - np.einsum("Iicdn,abdc->Iiabn", v_ref4, tei_tmp))
        #   sig(Iiab:babb) += -1.0*v(Ijcb:babb)*I(ajci:baba) + v(Ijca:babb)*I(bjci:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_4 = sig_4 - np.einsum("Ijcbn,ajci->Iiabn", v_ref4, tei_tmp) + np.einsum("Ijcan,bjci->Iiabn", v_ref4, tei_tmp)
        #   sig(Iiab:babb) += -1.0*v(Jiac:babb)*I(JbIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_4 = sig_4 - (np.einsum("Jiacn,JbIc->Iiabn", v_ref4, tei_tmp_J) - np.einsum("Jiacn,JbcI->Iiabn", v_ref4, tei_tmp_K))
        #   sig(Iiab:babb) += v(Jibc:babb)*I(JaIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_4 = sig_4 + (np.einsum("Jibcn,JaIc->Iiabn", v_ref4, tei_tmp_J) - np.einsum("Jibcn,JacI->Iiabn", v_ref4, tei_tmp_K))
        #   sig(Iiab:babb) += v(Jjab:babb)*I(JjIi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_4 = sig_4 + np.einsum("Jjabn,JjIi->Iiabn", v_ref4, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(4,5) v(5) = sig(4) (NC)
        ################################################ 

        #   sig(Iiab:babb) += v(jiAa:aaab)*I(jbAI:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 1)
        sig_4 = sig_4 - (np.einsum("Ajian,jbAI->Iiabn", v_ref5, tei_tmp) - np.einsum("Ajibn,jaAI->Iiabn", v_ref5, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(5,1) v(1) = sig(5)
        ################################################ 

        #   sig(ijAb:aaab) += v(jb:ab)*F(iA:aa) - v(ib:ab)*F(jA:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_5 = (np.einsum("jbn,iA->Aijbn", v_ref1, Fa_tmp) - np.einsum("ibn,jA->Aijbn", v_ref1, Fa_tmp))
        #   sig(ijAb:aaab) += - v(ic:ab)*I(Abjc:abab) + v(jc:ab)*I(Abic:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_5 = sig_5 - np.einsum("icn,Abjc->Aijbn", v_ref1, tei_tmp) + np.einsum("jcn,Abic->Aijbn", v_ref1, tei_tmp)
        #   sig(ijAb:aaab) += - v(kb:ab)*I(Akij:aaaa)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_5 = sig_5 - (np.einsum("kbn,Akij->Aijbn", v_ref1, tei_tmp) - np.einsum("kbn,Akji->Aijbn", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(5,2) v(2) = sig(5)
        ################################################ 

        #   sig(ijAb:aaab) += -v(Ib:ab)*I(AIij:aaaa)
        tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
        sig_5 = sig_5 - (np.einsum("Ibn,AIij->Aijbn", v_ref2, tei_tmp) - np.einsum("Ibn,AIji->Aijbn", v_ref2, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(5,3) v(3) = sig(5)
        ################################################ 

        #   sig(ijAb:aaab) += - v(jA:ab)*I(AbjC:abab) + v(jc:ab)*I(AbiC:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_5 = sig_5 - np.einsum("Cin,AbjC->Aijbn", v_ref3, tei_tmp)
        sig_5 = sig_5 + np.einsum("Cjn,AbiC->Aijbn", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(5,4) v(4) = sig(5)
        ################################################

        #   sig(ijAb:aaab) += - v(Iicb:babb)*I(Abjc:abab)
        tei_tmp = self.tei.get_subblock(3, 1, 2, 2)
        sig_5 = sig_5 - (np.einsum("Iicbn,AIjc->Aijbn", v_ref4, tei_tmp)
                         - np.einsum("Ijcbn,AIic->Aijbn", v_ref4, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(5,5) v(5) = sig(5)
        ################################################ 

        #   sig(ijAb:aaab) += t(ijAc:aaab)*F(bc:bb) + t(ijAb:aaab)*F(AB:bb) - t(ikAb:aaab)*F(jk:aa) + t(jkAb:aaab)*F(ik:aa)
        F_bc_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        F_AB_tmp = Fa[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_5 = sig_5 + np.einsum("Aijcn,bc->Aijbn", v_ref5, F_bc_tmp) # no contribution
        sig_5 = sig_5 + np.einsum("Bijbn,AB->Aijbn", v_ref5, F_AB_tmp) # no contribution
        Fi_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_5 = sig_5 - np.einsum("Aikbn,kj->Aijbn", v_ref5, Fi_tmp) + np.einsum("Ajkbn,ki->Aijbn", v_ref5, Fi_tmp)
        #   sig(ijAb:aaab) += v(ijBc:aaab)*I(abBc:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_5 = sig_5 + np.einsum("Bijcn,AbBc->Aijbn", v_ref5, tei_tmp)
        #   sig(ijAb:aaab) += 0.5*v(klAb:aaab)*I(klij:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_5 = sig_5 + 0.5*(np.einsum("Aklbn,klij->Aijbn", v_ref5, tei_tmp)
                             - np.einsum("Aklbn,klji->Aijbn", v_ref5, tei_tmp))
        #   sig(ijAb:aaab) += - v(kjAc:aaab)*I(kbic:abab)
        sig_5 = sig_5 - np.einsum("Akjcn,kbic->Aijbn", v_ref5, tei_tmp)
        #   sig(ijAb:aaab) += v(kiAc:aaab)*I(kbjc:abab)
        sig_5 = sig_5 + np.einsum("Akicn,kbjc->Aijbn", v_ref5, tei_tmp)
        #   sig(ijAb:aaab) += - v(ijCb:aaab)*I(AkCj:aaaa)
        tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
        sig_5 = sig_5 - (np.einsum("Cikbn,AkCj->Aijbn", v_ref5, tei_tmp_J) - np.einsum("Cikbn,AkjC->Aijbn", v_ref5, tei_tmp_K))
        #   sig(ijAb:aaab) += v(jkCb:aaab)*I(AkCi:aaaa)
        sig_5 = sig_5 + (np.einsum("Cjkbn,AkCi->Aijbn", v_ref5, tei_tmp_J) - np.einsum("Cjkbn,AkiC->Aijbn", v_ref5, tei_tmp_K))

        ################################################ 
        # sigs complete-- free to reshape!
        ################################################ 
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        sig_3 = np.reshape(sig_3, (v_b3.shape[0], v.shape[1]))
        # pack sig(4) vector for returning
        sig_4_out = np.zeros((v_b4.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for a in range(ras2):
                        for b in range(a):
                            sig_4_out[index, n] = sig_4[I, i, a, b, n]
                            index = index + 1
        # pack sig(5) vector for returning
        sig_5_out = np.zeros((v_b5.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for b in range(ras2):
                            sig_5_out[index, n] = sig_5[A, i, j, b, n]
                            index = index + 1

        # combine and return
        return np.vstack((sig_1, sig_2, sig_3, sig_4_out, sig_5_out)) + offset_v

    def do_cas_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do CAS-EA.

            block1 = v(a:b)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) = sig(1)
       
        """
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("bn,ab->an", v, F_tmp).reshape((v.shape[0], v.shape[1]))
        return sig_1 + offset_v

    def do_h_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do RAS(h)-EA.

            block1 = v(a:b)
            block2 = v(Iab:bbb)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) + | H(1,2) | * v(2) = sig(1)
            | H(2,1) | * v(1) + | H(2,2) | * v(2) = sig(2)
       
        """
        v_b1 = v[0:ras2, :] # v for block 1
        v_b2 = v[ras2:, :] # v for block 2
        # v(1) indexing: (a:b)
        v_ref1 = np.reshape(v_b1, (ras2, v.shape[1]))
        # v(2) indexing: (Iab:bbb)
        v_ref2 = np.zeros((ras1, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for a in range(ras2):
                    for b in range(a):
                        v_ref2[I, a, b, n] = v_b2[index, n]
                        v_ref2[I, b, a, n] = -1.0*v_b2[index, n]
                        index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(a:b) += v(a:b)*F(ab:bb)
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("bn,ab->an", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        #   sig(a:b) += v(a:b)*F(ab:bb)
        F_tmp = Fb[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("Iban,Ib->an", v_ref2, F_tmp)
        #   sig(a:b) += 0.5*v(Ibc:bbb)*I(Iabc:bbbb)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 + 0.5*(np.einsum("Ibcn,Iabc->an", v_ref2, tei_tmp) - np.einsum("Ibcn,Iacb->an", v_ref2, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        #   sig(Iab:bbb) += P(ab)*v(b:b)*F(aI:bb)
        F_tmp = Fb[ras1:ras1+ras2, 0:ras1]
        sig_2 = np.einsum("bn,aI->Iabn", v_ref1, F_tmp)
        sig_2 = sig_2 - np.einsum("an,bI->Iabn", v_ref1, F_tmp) #P(ab)
        #   sig(Iab:bbb) += v(c:b)*I(abIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
        sig_2 = sig_2 + (np.einsum("cn,abIc->Iabn", v_ref1, tei_tmp_J) - np.einsum("cn,abcI->Iabn", v_ref1, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        #   sig(Iab:bbb) += -v(Jab:bbb)*F(JI:bb)
        F_tmp = Fb[0:ras1, 0:ras1]
        sig_2 = sig_2 - np.einsum("Jabn,JI->Iabn", v_ref2, F_tmp)
        #   sig(Iab:bbb) += P(ab)*v(Icb:bbb)*F(ac:bb)
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + np.einsum("Icbn,ac->Iabn", v_ref2, F_tmp)
        sig_2 = sig_2 - np.einsum("Ican,bc->Iabn", v_ref2, F_tmp) #P(ab)
        #   sig(Iab:bbb) += -P(ab)*v(Jac:bbb)*I(JbIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 - (np.einsum("Jacn,JbIc->Iabn", v_ref2, tei_tmp_J) - np.einsum("Jacn,JbcI->Iabn", v_ref2, tei_tmp_K))
        sig_2 = sig_2 + (np.einsum("Jbcn,JaIc->Iabn", v_ref2, tei_tmp_J) - np.einsum("Jbcn,JacI->Iabn", v_ref2, tei_tmp_K)) #P(ab)
        #   sig(Iab:bbb) += 0.5*v(Icd:bbb)*I(abcd:bbbb)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_2 = sig_2 + 0.5*(np.einsum("Icdn,abcd->Iabn", v_ref2, tei_tmp) - np.einsum("Icdn,abdc->Iabn", v_ref2, tei_tmp))

        # Sigma evaluations done! Pack back up for returning
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2_out = np.zeros((v_b2.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for a in range(ras2):
                    for b in range(a):
                        sig_2_out[index, n] = sig_2[I, a, b, n]
                        index = index + 1

        return np.vstack((sig_1, sig_2_out)) + offset_v

    def do_p_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """Do RAS(p)-EA.

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
        nbf = ras1 + ras2 + ras3
        v_b1 = v[0:ras2, :] # v for block 1
        v_b2 = v[ras2:ras2+ras3, :] # v for block 2
        v_b3 = v[ras2+ras3:, :] # v for block 3
        # v(1) indexing: (a:b)
        v_ref1 = np.reshape(v_b1, (ras2, v.shape[1]))
        # v(2) indexing: (A:b)
        v_ref2 = np.reshape(v_b2, (ras3, v.shape[1]))
        # v(3) indexing: (iAa:aab)
        v_ref3 = np.reshape(v_b3, (ras3, ras2, ras2, v.shape[1]))

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = np.einsum("bn,ab->an", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        F_tmp = Fb[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("An,aA->an", v_ref2, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 
        F_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Aian,iA->an", v_ref3, F_tmp)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        sig_1 = sig_1 + np.einsum("Aibn,iaAb->an", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        F_tmp = Fb[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_2 = np.einsum("an,Aa->An", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        F_tmp = Fb[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_2 = sig_2 + np.einsum("Bn,AB->An", v_ref2, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 
        tei_tmp = self.tei.get_subblock(2, 3, 3, 2)
        sig_2 = sig_2 + np.einsum("Bian,iABa->An", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 
        F_tmp = Fa[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_3 = np.einsum("an,Ai->Aian", v_ref1, F_tmp)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_3 = sig_3 + np.einsum("bn,Aaib->Aian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 + np.einsum("Bn,AaiB->Aian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Ajan,ji->Aian", v_ref3, F_tmp)
        F_tmp = Fa[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_3 = sig_3 + np.einsum("Bian,AB->Aian", v_ref3, F_tmp)
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 + np.einsum("Aibn,ab->Aian", v_ref3, F_tmp)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("Ajbn,jaib->Aian", v_ref3, tei_tmp)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_3 = sig_3 + np.einsum("Bibn,AaBb->Aian", v_ref3, tei_tmp)
        tei_tmp_J = self.tei.get_subblock(3, 2, 2, 3)
        tei_tmp_K = self.tei.get_subblock(3, 2, 3, 2)
        sig_3 = sig_3 + (np.einsum("Bjan,AjiB->Aian", v_ref3, tei_tmp_J) - np.einsum("Bjan,AjBi->Aian", v_ref3, tei_tmp_K))

        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        sig_3 = np.reshape(sig_3, (v_b3.shape[0], v.shape[1]))

        return np.vstack((sig_1, sig_2, sig_3)) + offset_v

    def do_cas_ip(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do CAS-IP.

            block1 = v(i:a)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) = sig(1)
       
        """
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*np.einsum("jn,ji->in", v, F_tmp).reshape((v.shape[0], v.shape[1]))
        return sig_1 + offset_v

    def do_h_ip(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do RAS(h)-IP.

            block1 = v(i:a)
            block2 = v(I:a)
            block3 = v(Iia:bab)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) + | H(1,2) | * v(2) + | H(1,3) | * v(3) = sig(1)
            | H(2,1) | * v(1) + | H(2,2) | * v(2) + | H(2,3) | * v(3) = sig(2)
            | H(3,1) | * v(1) + | H(3,2) | * v(2) + | H(3,3) | * v(3) = sig(3)
       
        """
        v_b1 = v[0:ras2, :] # v for block 1
        v_b2 = v[ras2:ras1+ras2, :] # v for block 2
        v_b3 = v[ras1+ras2:, :] # v for block 3
        # v(1) indexing: (i:a)
        v_ref1 = np.reshape(v_b1, (ras2, v.shape[1]))
        # v(2) indexing: (I:a)
        v_ref2 = np.reshape(v_b2, (ras1, v.shape[1]))
        # v(3) indexing: (Iia:bab)
        v_ref3 = np.reshape(v_b3, (ras1, ras2, ras2, v.shape[1]))

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*np.einsum("jn,ji->in", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        F_tmp = Fa[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("In,Ii->in", v_ref2, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 
        F_tmp = Fb[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("Iian,Ia->in", v_ref3, F_tmp)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ijan,Ijai->in", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        F_tmp = Fa[ras1:ras1+ras2, 0:ras1]
        sig_2 = -1.0*np.einsum("in,iI->In", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        F_tmp = Fa[0:ras1, 0:ras1]
        sig_2 = sig_2 - np.einsum("Jn,JI->In", v_ref2, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 - np.einsum("Jian,JiaI->In", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 
        F_tmp = Fb[ras1:ras1+ras2, 0:ras1]
        sig_3 = np.einsum("in,aI->Iian", v_ref1, F_tmp)
        tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
        sig_3 = sig_3 - np.einsum("jn,ajIi->Iian", v_ref1, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################
        tei_tmp = self.tei.get_subblock(2, 1, 1, 2)
        sig_3 = sig_3 - np.einsum("Jn,aJIi->Iian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Ijan,ji->Iian", v_ref3, F_tmp)
        F_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 + np.einsum("Iibn,ab->Iian", v_ref3, F_tmp)
        F_tmp = Fb[0:ras1, 0:ras1]
        sig_3 = sig_3 - np.einsum("Jian,JI->Iian", v_ref3, F_tmp)

        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("Ijbn,ajbi->Iian", v_ref3, tei_tmp)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_3 = sig_3 + np.einsum("Jjan,JjIi->Iian", v_ref3, tei_tmp)
        tei_tmp_J = self.tei.get_subblock(2, 1, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 1, 2, 1)
        sig_3 = sig_3 + (np.einsum("Jibn,aJIb->Iian", v_ref3, tei_tmp_J)
                         - np.einsum("Jibn,aJbI->Iian", v_ref3, tei_tmp_K))

        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2 = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        sig_3 = np.reshape(sig_3, (v_b3.shape[0], v.shape[1]))

        return np.vstack((sig_1, sig_2, sig_3)) + offset_v

    def do_p_ip(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do RAS(p)-IP.

            block1 = v(i:a)
            block2 = v(ijA:aaa)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) + | H(1,2) | * v(2) = sig(1)
            | H(2,1) | * v(1) + | H(2,2) | * v(2) = sig(2)
       
        """
        v_b1 = v[0:ras2, :] # v for block 1
        v_b2 = v[ras2:, :] # v for block 2
        # v(1) indexing: (i:a)
        v_ref1 = np.reshape(v_b1, (ras2, v.shape[1]))
        # v(2) indexing: (Aij:aaa)
        v_ref2 = np.zeros((ras3, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        v_ref2[A, i, j, n] = v_b2[index, n]
                        v_ref2[A, j, i, n] = -1.0*v_b2[index, n]
                        index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(i:a) += -v(j:a)*F(ji:aa)
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*np.einsum("jn,ji->in", v_ref1, F_tmp)

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        #   sig(i:a) += v(jiA:aaa)*F(jA:aa)
        F_tmp = Fa[ras1:ras1+ras2, ras1+ras2:ras1+ras2+ras3]
        sig_1 = sig_1 + np.einsum("Ajin,jA->in", v_ref2, F_tmp)
        #   sig(i:a) += -0.5*v(jkA:aaa)*I(jkAi:aaaa)
        tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - 0.5*(np.einsum("Ajkn,jkAi->in", v_ref2, tei_tmp_J)
                             - np.einsum("Ajkn,jkiA->in", v_ref2, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        #   sig(ijA:aaa) += P(ij)*v(j:a)*F(Ai:aa)
        F_tmp = Fa[ras1+ras2:ras1+ras2+ras3, ras1:ras1+ras2]
        sig_2 = np.einsum("jn,Ai->Aijn", v_ref1, F_tmp)
        sig_2 = sig_2 - np.einsum("in,Aj->Aijn", v_ref1, F_tmp) #P(ij)
        #   sig(ijA:aaa) += -v(k:a)*I(Akij:aaaa)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_2 = sig_2 - (np.einsum("kn,Akij->Aijn", v_ref1, tei_tmp) - np.einsum("kn,Akji->Aijn", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        #   sig(ijA:aaa) += v(ijB:aaa)*F(AB:aa)
        F_tmp = Fa[ras1+ras2:ras1+ras2+ras3, ras1+ras2:ras1+ras2+ras3]
        sig_2 = sig_2 + np.einsum("Bijn,AB->Aijn", v_ref2, F_tmp)
        #   sig(ijA:aaa) += -P(ij)*v(kjA:aaa)*F(ki:aa)
        F_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 - np.einsum("Akjn,ki->Aijn", v_ref2, F_tmp)
        sig_2 = sig_2 + np.einsum("Akin,kj->Aijn", v_ref2, F_tmp) #P(ij)
        #   sig(ijA:aaa) += 0.5*v(klA:aaa)*I(klij:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_2 = sig_2 + 0.5*(np.einsum("Akln,klij->Aijn", v_ref2, tei_tmp) - np.einsum("Akln,klji->Aijn", v_ref2, tei_tmp))
        #   sig(ijA:aaa) += -P(ij)*v(ikB:aaa)*I(AkBj:aaaa)
        tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
        sig_2 = sig_2 - (np.einsum("Bikn,AkBj->Aijn", v_ref2, tei_tmp_J) - np.einsum("Bikn,AkjB->Aijn", v_ref2, tei_tmp_K))
        sig_2 = sig_2 + (np.einsum("Bjkn,AkBi->Aijn", v_ref2, tei_tmp_J) - np.einsum("Bjkn,AkiB->Aijn", v_ref2, tei_tmp_K)) #P(ij)

        # Sigma evaluations done! Pack back up for returning
        sig_1 = np.reshape(sig_1, (v_b1.shape[0], v.shape[1]))
        sig_2_out = np.zeros((v_b2.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        sig_2_out[index, n] = sig_2[A, i, j, n]
                        index = index + 1

        return np.vstack((sig_1, sig_2_out)) + offset_v

    def do_cas_1sf_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do CAS-1SF-EA.

            block1 = v(ija:aab)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) = sig(1)
       
        """
        # v(1) unpack to indexing: (ija:aab)
        v_ref1 = np.zeros((ras2,ras2,ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        v_ref1[i, a, b, n] = v[index, n]
                        v_ref1[i, b, a, n] = -1.0*v[index, n]
                        index = index + 1 
        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ija:aab) += -P(ab)*t(iab:abb)*F(ac:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = (np.einsum("icbn,ac->iabn", v_ref1, Fb_tmp) - np.einsum("ican,bc->iabn", v_ref1, Fb_tmp))
        #   sig(ija:aab) += t(jab:abb)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("jabn,ji->iabn", v_ref1, Fa_tmp)
        #   sig(ija:aab) += P(ab)*t(jac:abb)*I(jbic:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - (np.einsum("jacn,jbic->iabn", v_ref1, tei_tmp) - np.einsum("jbcn,jaic->iabn", v_ref1, tei_tmp))
        #   sig(ija:aab) += -0.5*t(icd:abb)*I(abcd:abab)
        sig_1 = sig_1 + 0.5*(np.einsum("icdn,abcd->iabn", v_ref1, tei_tmp) - np.einsum("icdn,abdc->iabn", v_ref1, tei_tmp))

        sig_1_out = np.zeros((v.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        sig_1_out[index] = sig_1[i, a, b]
                        index = index + 1

        return sig_1_out + offset_v

    def do_cas_1sf_ip(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do CAS-1SF-IP.

            block1 = v(ija:aab)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) = sig(1)
       
        """
        # v(1) unpack to indexing: (ija:aab)
        v_ref1 = np.zeros((ras2,ras2,ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        v_ref1[i, j, a, n] = v[index, n]
                        v_ref1[j, i, a, n] = -1.0*v[index, n]
                        index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ija:aab) += -P(ij)*v(kja:aab)*F(ki:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*(np.einsum("kjan,ki->ijan", v_ref1, Fa_tmp) - np.einsum("kian,kj->ijan", v_ref1, Fa_tmp))
        #   sig(ija:aab) += v(ijb:aab)*F(ab:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("ijbn,ab->ijan", v_ref1, Fb_tmp)
        #   sig(ija:aab) += -P(ij)*v(ikb:aab)*I(akbj:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - (np.einsum("ikbn,akbj->ijan", v_ref1, tei_tmp) - np.einsum("jkbn,akbi->ijan", v_ref1, tei_tmp))
        #   sig(ija:aab) += 0.5*v(kla:aab)*I(klij:aaaa)
        sig_1 = sig_1 + 0.5*(np.einsum("lkan,lkij->ijan", v_ref1, tei_tmp) - np.einsum("lkan,lkji->ijan", v_ref1, tei_tmp))

        sig_1_out = np.zeros((v.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0 
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        sig_1_out[index, n] = sig_1[i, j, a, n]
                        index = index + 1 

        return sig_1_out + offset_v

    def do_h_1sf_ip(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do RAS(h)-1SF-IP.

            block1 = v(ija:aab)
            block2 = v(Iia:aab)
            block2 = v(Iijab:baabb)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) + | H(1,2) | * v(2) + | H(1,3) | * v(3) = sig(1)
            | H(2,1) | * v(1) + | H(2,2) | * v(2) + | H(2,3) | * v(3) = sig(2)
            | H(3,1) | * v(1) + | H(3,2) | * v(2) + | H(3,3) | * v(3) = sig(3)
       
        """
        nbf = ras1 + ras2 + ras3

        n_b1_dets = int(ras2 * ((ras2-1)*(ras2)/2))
        n_b2_dets = int(ras2 * ras1 * ras2)
        v_b1 = v[0:n_b1_dets, :]
        v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets, :]
        v_b3 = v[n_b1_dets+n_b2_dets:, :]

        # v(1) unpack to indexing: (ija:aab)
        v_ref1 = np.zeros((ras2,ras2,ras2,v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        v_ref1[i, j, a, n] = v_b1[index, n]
                        v_ref1[j, i, a, n] = -1.0*v_b1[index, n]
                        index = index + 1
        # v(2) unpack to indexing: (Iia:aab)
        v_ref2 = np.reshape(v_b2, (ras1, ras2, ras2, v.shape[1]))
        # v(3) unpack to indexing: (Iijab:aaabb)
        v_ref3 = np.zeros((ras1, ras2, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for j in range(i):
                        for a in range(ras2):
                            for b in range(a):
                                v_ref3[I, i, j, a, b, n] = v_b3[index, n]
                                v_ref3[I, j, i, a, b, n] = -1.0*v_b3[index, n]
                                v_ref3[I, i, j, b, a, n] = -1.0*v_b3[index, n]
                                v_ref3[I, j, i, b, a, n] = v_b3[index, n]
                                index = index + 1

        ################################################ 
        # Do the following term:
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(ija:aab) += -P(ij)*v(kja:aab)*F(ki:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = -1.0*np.einsum("kjan,ki->ijan", v_ref1, Fa_tmp)
        sig_1 = sig_1 + np.einsum("kian,kj->ijan", v_ref1, Fa_tmp) #P(ij)
        #   sig(ija:aab) += v(ijb:aab)*F(ab:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("ijbn,ab->ijan", v_ref1, Fb_tmp)
        #   sig(ija:aab) += -P(ij)*v(ikb:aab)*I(akbj:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("ikbn,akbj->ijan", v_ref1, tei_tmp)
        sig_1 = sig_1 + np.einsum("jkbn,akbi->ijan", v_ref1, tei_tmp) #P(ij)
        #   sig(ija:aab) += 0.5*v(kla:aab)*I(klij:aaaa)
        sig_1 = sig_1 + 0.5*(np.einsum("lkan,lkij->ijan", v_ref1, tei_tmp) - np.einsum("lkan,lkji->ijan", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        #   sig(ija:aab) += -P(ij)*v(Ija:aab)*F(Ii:aa)
        Fa_tmp = Fa[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("Ijan,Ii->ijan", v_ref2, Fa_tmp)
        sig_1 = sig_1 + np.einsum("Iian,Ij->ijan", v_ref2, Fa_tmp) #P(ij)
        #   sig(ija:aab) += v(Ika:aab)*F(Ikij:aaaa)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 + (np.einsum("Ikan,Ikij->ijan", v_ref2, tei_tmp) - np.einsum("Ikan,Ikji->ijan", v_ref2, tei_tmp))
        #   sig(ija:aab) += -P(ij)*v(Ijb:aab)*F(aIbi:baba)
        tei_tmp = self.tei.get_subblock(2, 1, 2, 2)
        sig_1 = sig_1 - np.einsum("Ijbn,aIbi->ijan", v_ref2, tei_tmp)
        sig_1 = sig_1 + np.einsum("Iibn,aIbj->ijan", v_ref2, tei_tmp) #P(ij)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 
        #   sig(Iia:aab) += v(Iijba:baabb)*F(Ib:bb)
        Fb_tmp = Fb[0:ras1, ras1:ras1+ras2]
        sig_1 = sig_1 + np.einsum("Iijban,Ib->ijan", v_ref3, Fb_tmp)
        #   sig(Iia:aab) += v(Iijbc:baabb)*F(Iabc:bbbb)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 + 0.5*(np.einsum("Iijbcn,Iabc->ijan", v_ref3, tei_tmp) - np.einsum("Iijbcn,Iacb->ijan", v_ref3, tei_tmp))
        #   sig(Iia:aab) += -P(ij)*v(Ikjba:baabb)*F(Ikbi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 2)
        sig_1 = sig_1 - np.einsum("Ikjban,Ikbi->ijan", v_ref3, tei_tmp)
        sig_1 = sig_1 + np.einsum("Ikiban,Ikbj->ijan", v_ref3, tei_tmp) #P(ij)

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        #   sig(Iia:aab) += -1.0*v(jia:aab)*F(Ij:aa)
        Fa_tmp = Fa[0:ras1, ras1:ras1+ras2]
        sig_2 = -1.0*np.einsum("jian,Ij->Iian", v_ref1, Fa_tmp)
        #   sig(ija:aab) += -1.0*v(jib:aab)*I(ajbI:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 1)
        sig_2 = sig_2 - np.einsum("jibn,ajbI->Iian", v_ref1, tei_tmp) 
        #   sig(ija:aab) += 0.5*v(jka:aab)*I(jkIi:aaaa)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
        sig_2 = sig_2 + 0.5*(np.einsum("jkan,jkIi->Iian", v_ref1, tei_tmp_J)
                             - np.einsum("jkan,jkiI->Iian", v_ref1, tei_tmp_K))

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        #   sig(Iia:aab) += -1.0*v(Jia:aab)*F(JI:aa)
        Fa_tmp = Fa[0:ras1, 0:ras1]
        sig_2 = sig_2 - np.einsum("Jian,JI->Iian", v_ref2, Fa_tmp)
        #   sig(Iia:aab) += -1.0*v(Jia:aab)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 - np.einsum("Ijan,ji->Iian", v_ref2, Fa_tmp)
        #   sig(Iia:aab) += v(Iib:aab)*F(ab:aa)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + np.einsum("Iibn,ab->Iian", v_ref2, Fb_tmp)
        #   sig(Iia:aab) += v(Jja:aab)*I(JjIi:aaaa)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 + (np.einsum("Jjan,JjIi->Iian", v_ref2, tei_tmp_J)
                         - np.einsum("Jjan,JjiI->Iian", v_ref2, tei_tmp_K))
        #   sig(Iia:aab) += -1.0*v(Ijb:aab)*I(ajbi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_2 = sig_2 - np.einsum("Ijbn,ajbi->Iian", v_ref2, tei_tmp)
        #   sig(Iia:aab) += -1.0*v(Jib:aab)*I(aJbI:baba)
        tei_tmp = self.tei.get_subblock(2, 1, 2, 1)
        sig_2 = sig_2 - np.einsum("Jibn,aJbI->Iian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 
        #   sig(Iia:aab) += -1.0*v(Jjiba:baabb)*F(JjbI:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 2, 1)
        sig_2 = sig_2 - np.einsum("Jjiban,JjbI->Iian", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 
        #   sig(Iijab:baabb) += P(ab)*v(ijb:aab)*F(aI:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, 0:ras1]
        sig_3 = np.einsum("ijbn,aI->Iijabn", v_ref1, Fb_tmp)
        sig_3 = sig_3 - np.einsum("ijan,bI->Iijabn", v_ref1, Fb_tmp) #P(ab)
        #   sig(Iijab:baabb) += v(ijc:aab)*I(abIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 1)
        sig_3 = sig_3 + (np.einsum("ijcn,abIc->Iijabn", v_ref1, tei_tmp_J)
                         - np.einsum("ijcn,abcI->Iijabn", v_ref1, tei_tmp_K))
        #   sig(Iijab:baabb) += -1.0*P(ij)*P(ab)*v(kjb:aab)*I(akIi:bbbb)
        tei_tmp = self.tei.get_subblock(2, 2, 1, 2)
        sig_3 = sig_3 - np.einsum("kjbn,akIi->Iijabn", v_ref1, tei_tmp)
        sig_3 = sig_3 + np.einsum("kibn,akIj->Iijabn", v_ref1, tei_tmp) #P(ij)
        sig_3 = sig_3 + np.einsum("kjan,bkIi->Iijabn", v_ref1, tei_tmp) #P(ab)
        sig_3 = sig_3 - np.einsum("kian,bkIj->Iijabn", v_ref1, tei_tmp) #P(ij)P(ab)

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 
        #   sig(Iijab:baabb) += -1.0*P(ij)*v(Jjb:aab)*I(aJIi:baba)
        tei_tmp = self.tei.get_subblock(2, 1, 1, 2)
        sig_3 = sig_3 - np.einsum("Jjbn,aJIi->Iijabn", v_ref2, tei_tmp)
        sig_3 = sig_3 + np.einsum("Jibn,aJIj->Iijabn", v_ref2, tei_tmp) #P(ij)
        sig_3 = sig_3 + np.einsum("Jjan,bJIi->Iijabn", v_ref2, tei_tmp) #P(ab)
        sig_3 = sig_3 - np.einsum("Jian,bJIj->Iijabn", v_ref2, tei_tmp) #P(ij)P(ab)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 
        #   sig(Iijab:baabb) += P(ab)*v(Iijcb:baabb)*F(ac:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 + np.einsum("Iijcbn,ac->Iijabn", v_ref3, Fb_tmp)
        sig_3 = sig_3 - np.einsum("Iijcan,bc->Iijabn", v_ref3, Fb_tmp) #P(ab)
        #   sig(Iijab:baabb) += -1.0*v(Jijab:baabb)*F(JI:bb)
        Fb_tmp = Fb[0:ras1, 0:ras1]
        sig_3 = sig_3 - np.einsum("Jijabn,JI->Iijabn", v_ref3, Fb_tmp)
        #   sig(Iijab:baabb) += -P(ij)*v(Ikjab:baabb)*F(ki:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Ikjabn,ki->Iijabn", v_ref3, Fa_tmp)
        sig_3 = sig_3 + np.einsum("Ikiabn,kj->Iijabn", v_ref3, Fa_tmp) #P(ij)
        #   sig(Iijab:baabb) += P(ij)*v(Jkjab:baabb)*I(JkIi:baba)
        tei_tmp = self.tei.get_subblock(1, 2, 1, 2)
        sig_3 = sig_3 + np.einsum("Jkjabn,JkIi->Iijabn", v_ref3, tei_tmp)
        sig_3 = sig_3 - np.einsum("Jkiabn,JkIj->Iijabn", v_ref3, tei_tmp) #P(ij)
        #   sig(Iijab:baabb) += -P(ab)*v(Jijac:baabb)*I(JbIc:bbbb)
        tei_tmp_J = self.tei.get_subblock(1, 2, 1, 2)
        tei_tmp_K = self.tei.get_subblock(1, 2, 2, 1)
        sig_3 = sig_3 - (np.einsum("Jijacn,JbIc->Iijabn", v_ref3, tei_tmp_J)
                         - np.einsum("Jijacn,JbcI->Iijabn", v_ref3, tei_tmp_K))
        sig_3 = sig_3 + (np.einsum("Jijbcn,JaIc->Iijabn", v_ref3, tei_tmp_J)
                         - np.einsum("Jijbcn,JacI->Iijabn", v_ref3, tei_tmp_K)) #P(ab)
        #   sig(Iijab:baabb) += -P(ab)*P(ij)*v(Ikjcb:baabb)*I(ciak:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("Ikjcbn,ciak->Iijabn", v_ref3, tei_tmp)
        sig_3 = sig_3 + np.einsum("Ikicbn,cjak->Iijabn", v_ref3, tei_tmp) #P(ij)
        sig_3 = sig_3 + np.einsum("Ikjcan,cibk->Iijabn", v_ref3, tei_tmp) #P(ab)
        sig_3 = sig_3 - np.einsum("Ikican,cjbk->Iijabn", v_ref3, tei_tmp) #P(ij)P(ab)
        #   sig(Iijab:baabb) += 0.5*v(Iijcd:baabb)*I(abcd:bbbb)
        sig_3 = sig_3 + 0.5*(np.einsum("Iijcdn,abcd->Iijabn", v_ref3, tei_tmp)
                             - np.einsum("Iijcdn,abdc->Iijabn", v_ref3, tei_tmp))
        #   sig(Iijab:baabb) += 0.5*v(Iklab:baabb)*I(klij:bbbb)
        sig_3 = sig_3 + 0.5*(np.einsum("Iklabn,klij->Iijabn", v_ref3, tei_tmp)
                             - np.einsum("Iklabn,klji->Iijabn", v_ref3, tei_tmp))

        sig_1_out = np.zeros((v_b1.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for a in range(ras2):
                        sig_1_out[index, n] = sig_1[i, j, a, n]
                        index = index + 1
        # v(2) repack
        sig_2_out = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        # v(3) repack
        sig_3_out = np.zeros((v_b3.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for I in range(ras1):
                for i in range(ras2):
                    for j in range(i):
                        for a in range(ras2):
                            for b in range(a):
                                sig_3_out[index, n] = sig_3[I, i, j, a, b, n]
                                index = index + 1

        return np.vstack((sig_1_out, sig_2_out, sig_3_out)) + offset_v

    '''
    def do_h_1sf_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do RAS(h)-1SF-EA.
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
        n_b1_dets = int(ras2 * ((ras2-1)*(ras2)/2)) 
        n_b2_dets = int(ras1 * ((ras2-1)*(ras2)/2))
        v_b1 = v[0:n_b1_dets]
        v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets]
        v_b3 = v[n_b1_dets+n_b2_dets:]
        # v(1) unpack to indexing: (iab:abb)
        v_ref1 = np.zeros((ras2,ras2,ras2))
        index = 0
        for i in range(ras2):
            for a in range(ras2):
                for b in range(a):
                    v_ref1[i, a, b] = v_b1[index]
                    v_ref1[i, b, a] = -1.0*v_b1[index]
                    index = index + 1
        # v(2) unpack to indexing: (Iab:abb)
        v_ref2 = np.zeros((ras1,ras2,ras2))
        index = 0
        for I in range(ras1):
            for a in range(ras2):
                for b in range(a):
                    v_ref2[I, a, b] = v_b2[index]
                    v_ref2[I, b, a] = -1.0*v_b2[index]
                    index = index + 1
        # v(3) unpack to indexing: (Iiabc:babbb)
        v_ref3 = np.zeros((ras1,ras2,ras2,ras2,ras2))
        index = 0
        for I in range(ras1):
            for i in range(ras2):
                for a in range(ras2):
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
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = (np.einsum("icb,ac->iab", v_ref1, Fb_tmp) - np.einsum("ica,bc->iab", v_ref1, Fb_tmp))
        #   sig(ija:aab) += -t(jab:abb)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
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
        Fa_tmp = Fa[0:ras1, ras1:ras1+ras2]
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
        Fb_tmp = Fb[0:ras1, ras1:ras1+ras2]
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
        Fa_tmp = Fa[ras1:ras1+ras2, 0:ras1]
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
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + (np.einsum("Icb,ac->Iab", v_ref2, Fb_tmp) - np.einsum("Ica,bc->Iab", v_ref2, Fb_tmp))
        #   sig(Iab:abb) += -t(Jab:abb)*F(JI:aa)
        Fa_tmp = Fa[0:ras1, 0:ras1]
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
        Fb_tmp = Fb[ras1:ras1+ras2, 0:ras1]
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
        Fb_tmp = Fb[ras1:ras1+ras2, 0:ras1]
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
        Fb_tmp = Fb[0:ras1, 0:ras1]
        sig_3 = sig_3 - np.einsum("Jiabc,JI->Iiabc", v_ref3, Fb_tmp)
        #   sig(Iiabc:babbb) += -t(Ijabc:babbb)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Ijabc,ji->Iiabc", v_ref3, Fa_tmp)
        #   sig(Iiabc:babbb) += P(ab)*P(ac)*t(Iidbc:babbb)*F(ad:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
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

        sig_1_out = np.zeros((v_b1.shape[0], 1))
        index = 0
        for i in range(ras2):
            for a in range(ras2):
                for b in range(a):
                    if(abs(sig_1[i, a, b] + sig_1[i, b, a]) > 1e-10):
                        print("ERR: REF 1 NOT ANTISYMMETRIC")
                    sig_1_out[index] = sig_1[i, a, b]
                    index = index + 1
        sig_2_out = np.zeros((v_b2.shape[0], 1))
        index = 0
        for I in range(ras1):
            for a in range(ras2):
                for b in range(a):
                    if(abs(sig_2[I, a, b] + sig_2[I, b, a]) > 1e-10):
                        print("ERR: REF 2 NOT ANTISYMMETRIC")
                    sig_2_out[index] = sig_2[I, a, b]
                    index = index + 1
        sig_3_out = np.zeros((v_b3.shape[0], 1))
        index = 0
        for I in range(ras1):
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        for c in range(b):
                            if(abs(sig_3[I, i, a, b, c] + sig_3[I, i, b, a, c]) > 1e-10):
                                print("ERR: REF 3 NOT ANTISYMMETRIC")
                            if(abs(sig_3[I, i, a, b, c] + sig_3[I, i, a, c, b]) > 1e-10):
                                print("ERR: REF 3 NOT ANTISYMMETRIC")
                            sig_3_out[index] = v_ref3[I, i, a, b, c]
                            index = index + 1

        return np.vstack((sig_1_out, sig_2_out, sig_3_out)) + offset_v
    '''

    def do_p_1sf_ea(self, v, Fa, Fb, tei, offset_v, ras1, ras2, ras3):
        """ Do RAS(p)-1SF-EA.

            block1 = v(iab:abb)
            block2 = v(Iab:abb)
            block2 = v(Iiabc:babbb)

            Evaluate the following matrix vector multiply:

            | H(1,1) | * v(1) = sig(1)
       
        """
        nbf = ras1 + ras2 + ras3

        n_b1_dets = int(ras2 * ((ras2-1)*(ras2)/2))
        n_b2_dets = int(ras2 * ras3 * ras2)
        v_b1 = v[0:n_b1_dets, :]
        v_b2 = v[n_b1_dets:n_b1_dets+n_b2_dets, :]
        v_b3 = v[n_b1_dets+n_b2_dets:, :]

        # v(1) unpack to indexing: (iab:abb)
        v_ref1 = np.zeros((ras2,ras2,ras2,v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        v_ref1[i, a, b, n] = v_b1[index, n]
                        v_ref1[i, b, a, n] = -1.0*v_b1[index, n]
                        index = index + 1
        # v(2) unpack to indexing: (Aia:bab)
        v_ref2 = np.reshape(v_b2, (ras3, ras2, ras2, v.shape[1]))
        # v(3) unpack to indexing: (Aijab:baabb)
        v_ref3 = np.zeros((ras3, ras2, ras2, ras2, ras2, v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for a in range(ras2):
                            for b in range(a):
                                v_ref3[A, i, j, a, b, n] = v_b3[index, n]
                                v_ref3[A, j, i, a, b, n] = -1.0*v_b3[index, n]
                                v_ref3[A, i, j, b, a, n] = -1.0*v_b3[index, n]
                                v_ref3[A, j, i, b, a, n] = v_b3[index, n]
                                index = index + 1

        ################################################ 
        # Do the following term: OK
        #       H(1,1) v(1) = sig(1)
        ################################################ 
        #   sig(iab:abb) += P(ab)*t(iab:abb)*F(ac:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = (np.einsum("icbn,ac->iabn", v_ref1, Fb_tmp) - np.einsum("ican,bc->iabn", v_ref1, Fb_tmp))
        #   sig(iab:abb) += -t(jab:abb)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_1 = sig_1 - np.einsum("jabn,ji->iabn", v_ref1, Fa_tmp)
        #   sig(iab:abb) += -P(ab)*t(jac:abb)*I(jbic:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_1 = sig_1 - (np.einsum("jacn,jbic->iabn", v_ref1, tei_tmp) - np.einsum("jbcn,jaic->iabn", v_ref1, tei_tmp))
        #   sig(iab:abb) += 0.5*t(icd:abb)*I(abcd:abab)
        sig_1 = sig_1 + 0.5*(np.einsum("icdn,abcd->iabn", v_ref1, tei_tmp) - np.einsum("icdn,abdc->iabn", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(1,2) v(2) = sig(1)
        ################################################ 
        #   sig(iab:abb) += t(iBb:abb)*F(aB:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Bibn,aB->iabn", v_ref2, Fb_tmp)
        sig_1 = sig_1 - np.einsum("Bian,bB->iabn", v_ref2, Fb_tmp) #P(ab)
        #   sig(iab:abb) += t(iAd:abb)*I(abAd:bbbb)
        tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 + (np.einsum("Aidn,abAd->iabn", v_ref2, tei_tmp_J) - np.einsum("Aidn,abdA->iabn", v_ref2, tei_tmp_K))
        #   sig(iab:abb) += -t(jAb:abb)*I(ajAi:baba)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        sig_1 = sig_1 - (np.einsum("Ajbn,ajAi->iabn", v_ref2, tei_tmp))
        sig_1 = sig_1 + (np.einsum("Ajan,bjAi->iabn", v_ref2, tei_tmp)) #P(ab)

        ################################################ 
        # Do the following term:
        #       H(1,3) v(3) = sig(1)
        ################################################ 
        # TESTING
        #   sig(iab:abb) += t(jiAab:aabbb)*F(jA:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1+ras2:nbf]
        sig_1 = sig_1 + np.einsum("Ajiabn,jA->iabn", v_ref3, Fa_tmp)
        #   sig(iab:abb) += -1.0*t(jkAab:aabbb)*I(jkAi:aaaa)
        tei_tmp_J = self.tei.get_subblock(2, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(2, 2, 2, 3)
        sig_1 = sig_1 - 0.5*(np.einsum("Ajkabn,jkAi->iabn", v_ref3, tei_tmp_J)
                             - np.einsum("Ajkabn,jkiA->iabn", v_ref3, tei_tmp_K))
        #   sig(iab:abb) += t(jiAcb:aabbb)*I(jaAc:aaaa)
        tei_tmp = self.tei.get_subblock(2, 2, 3, 2)
        sig_1 = sig_1 + np.einsum("Ajicbn,jaAc->iabn", v_ref3, tei_tmp)
        sig_1 = sig_1 - np.einsum("Ajican,jbAc->iabn", v_ref3, tei_tmp) #P(ab)

        ################################################ 
        # Do the following term:
        #       H(2,1) v(1) = sig(2)
        ################################################ 
        #   sig(iAa:abb) += t(iba:abb)*F(Ab:bb)
        Fb_tmp = Fb[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_2 = np.einsum("iban,Ab->Aian", v_ref1, Fb_tmp)
        #   sig(iAa:abb) += 0.5*t(ibc:abb)*I(Aabc:bbbb)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_2 = sig_2 + 0.5*(np.einsum("ibcn,Aabc->Aian", v_ref1, tei_tmp)
                             - np.einsum("ibcn,Aacb->Aian", v_ref1, tei_tmp))
        #   sig(iab:abb) += -t(jab:abb)*I(Ajai:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_2 = sig_2 - (np.einsum("jban,Ajbi->Aian", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(2,2) v(2) = sig(2)
        ################################################ 
        #   sig(iAa:abb) += t(iBa:abb)*F(AB:bb)
        Fb_tmp = Fb[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_2 = sig_2 + np.einsum("Bian,AB->Aian", v_ref2, Fb_tmp)
        #   sig(iAa:abb) += -t(jAa:abb)*F(ji:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 - np.einsum("Ajan,ji->Aian", v_ref2, Fa_tmp)
        #   sig(iAa:abb) += -t(jAa:abb)*F(ji:aa)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_2 = sig_2 + np.einsum("Aibn,ab->Aian", v_ref2, Fb_tmp)
        #   sig(iAa:abb) += t(iBb:abb)*I(AaBb:bbbb)
        tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
        sig_2 = sig_2 + (np.einsum("Bibn,AaBb->Aian", v_ref2, tei_tmp_J)
                         - np.einsum("Bibn,AabB->Aian", v_ref2, tei_tmp_K))
        #   sig(iAa:abb) += -1.0*t(jAb:abb)*I(jaib:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_2 = sig_2 - np.einsum("Ajbn,jaib->Aian", v_ref2, tei_tmp)
        #   sig(iAa:abb) += -1.0*t(jBb:abb)*I(AjBi:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_2 = sig_2 - np.einsum("Bjan,AjBi->Aian", v_ref2, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(2,3) v(3) = sig(2)
        ################################################ 
        #   sig(iAb:abb) += t(jiBba:aabbb)*I(jABb:aaaa)
        tei_tmp = self.tei.get_subblock(2, 3, 3, 2)
        sig_2 = sig_2 + np.einsum("Bjiban,jABb->Aian", v_ref3, tei_tmp)

        ################################################ 
        # Do the following term:
        #       H(3,1) v(1) = sig(3)
        ################################################ 
        #   sig(ijAab:aaabb) += t(jab:abb)*F(Ai:aa)
        Fa_tmp = Fa[ras1+ras2:nbf, ras1:ras1+ras2]
        sig_3 = np.einsum("jabn,Ai->Aijabn", v_ref1, Fa_tmp)
        sig_3 = sig_3 - np.einsum("iabn,Aj->Aijabn", v_ref1, Fa_tmp) #P(ij)
        #   sig(ijAab:aaabb) += P(ab)*t(jcb:abb)*I(Aaic:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 2)
        sig_3 = sig_3 + np.einsum("jcbn,Aaic->Aijabn", v_ref1, tei_tmp)
        sig_3 = sig_3 - np.einsum("jcan,Abic->Aijabn", v_ref1, tei_tmp) #P(ab)
        sig_3 = sig_3 - np.einsum("icbn,Aajc->Aijabn", v_ref1, tei_tmp) #P(ij)
        sig_3 = sig_3 + np.einsum("ican,Abjc->Aijabn", v_ref1, tei_tmp) #P(ab)P(ij)
        #   sig(ijAab:aaabb) += -1.0*t(kab:abb)*I(Akij:aaaa)
        sig_3 = sig_3 - (np.einsum("kabn,Akij->Aijabn", v_ref1, tei_tmp)
                         - np.einsum("kabn,Akji->Aijabn", v_ref1, tei_tmp))

        ################################################ 
        # Do the following term:
        #       H(3,2) v(2) = sig(3)
        ################################################ 
        #   sig(ijAab:aaabb) += P(ab)*t(jBb:abb)*I(AaiB:abab)
        tei_tmp = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 + np.einsum("Bjbn,AaiB->Aijabn", v_ref2, tei_tmp)
        sig_3 = sig_3 - np.einsum("Bjan,AbiB->Aijabn", v_ref2, tei_tmp) #P(ab)
        sig_3 = sig_3 - np.einsum("Bibn,AajB->Aijabn", v_ref2, tei_tmp) #P(ij)
        sig_3 = sig_3 + np.einsum("Bian,AbjB->Aijabn", v_ref2, tei_tmp) #P(ab)P(ij)

        ################################################ 
        # Do the following term:
        #       H(3,3) v(3) = sig(3)
        ################################################ 
        #   sig(ijAab:aaabb) += -1.0*P(ij)*t(kjAab:aabbb)*F(ki:aa)
        Fa_tmp = Fa[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 - np.einsum("Akjabn,ki->Aijabn", v_ref3, Fa_tmp)
        sig_3 = sig_3 + np.einsum("Akiabn,kj->Aijabn", v_ref3, Fa_tmp) #P(ij)
        #   sig(ijAab:aaabb) += t(ijBab:aabbb)*F(AB:aa)
        Fa_tmp = Fa[ras1+ras2:nbf, ras1+ras2:nbf]
        sig_3 = sig_3 + np.einsum("Bijabn,AB->Aijabn", v_ref3, Fa_tmp)
        #   sig(ijAab:aaabb) += P(ab)*t(ijAcb:aabbb)*F(ac:bb)
        Fb_tmp = Fb[ras1:ras1+ras2, ras1:ras1+ras2]
        sig_3 = sig_3 + np.einsum("Aijcbn,ac->Aijabn", v_ref3, Fb_tmp)
        sig_3 = sig_3 - np.einsum("Aijcan,bc->Aijabn", v_ref3, Fb_tmp) #P(ab)
        #   sig(ijAab:aaabb) += P(ab)*t(ijBcb:aabbb)*I(AaBc:baba)
        tei_tmp = self.tei.get_subblock(3, 2, 3, 2)
        sig_3 = sig_3 + np.einsum("Bijcbn,AaBc->Aijabn", v_ref3, tei_tmp) 
        sig_3 = sig_3 - np.einsum("Bijcan,AbBc->Aijabn", v_ref3, tei_tmp) #P(ab)
        #   sig(ijAab:aaabb) += -1.0*P(ij)*t(ikBab:aabbb)*I(AkBj:aaaa)
        tei_tmp_J = self.tei.get_subblock(3, 2, 3, 2)
        tei_tmp_K = self.tei.get_subblock(3, 2, 2, 3)
        sig_3 = sig_3 - (np.einsum("Bikabn,AkBj->Aijabn", v_ref3, tei_tmp_J)
                         - np.einsum("Bikabn,AkjB->Aijabn", v_ref3, tei_tmp_K))
        sig_3 = sig_3 + (np.einsum("Bjkabn,AkBi->Aijabn", v_ref3, tei_tmp_J)
                         - np.einsum("Bjkabn,AkiB->Aijabn", v_ref3, tei_tmp_K))
        #   sig(ijAab:aaabb) += -1.0*P(ij)*P(ab)*t(kjAcb:aabbb)*I(kaic:abab)
        tei_tmp = self.tei.get_subblock(2, 2, 2, 2)
        sig_3 = sig_3 - np.einsum("Akjcbn,kaic->Aijabn", v_ref3, tei_tmp) 
        sig_3 = sig_3 + np.einsum("Akjcan,kbic->Aijabn", v_ref3, tei_tmp) #P(ab)
        sig_3 = sig_3 + np.einsum("Akicbn,kajc->Aijabn", v_ref3, tei_tmp) #P(ij)
        sig_3 = sig_3 - np.einsum("Akican,kbjc->Aijabn", v_ref3, tei_tmp) #P(ab)P(ij)
        #   sig(ijAab:aaabb) += 0.5*t(klAab:aabbb)*I(klij:aaaa)
        sig_3 = sig_3 + 0.5*(np.einsum("Aklabn,klij->Aijabn", v_ref3, tei_tmp)
                             - np.einsum("Aklabn,klji->Aijabn", v_ref3, tei_tmp))
        #   sig(ijAab:aaabb) += 0.5*t(ijAcd:aabbb)*I(abcd:bbbb)
        sig_3 = sig_3 + 0.5*(np.einsum("Aijcdn,abcd->Aijabn", v_ref3, tei_tmp)
                             - np.einsum("Aijcdn,abdc->Aijabn", v_ref3, tei_tmp))

        # v(1) unpack to indexing: (iab:abb)
        sig_1_out = np.zeros((v_b1.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for a in range(ras2):
                    for b in range(a):
                        sig_1_out[index, n] = sig_1[i, a, b, n]
                        index = index + 1
        # v(2) unpack to indexing: (iAa:abb)
        sig_2_out = np.reshape(sig_2, (v_b2.shape[0], v.shape[1]))
        # v(3) unpack to indexing: (ijAab:aaabb)
        sig_3_out = np.zeros((v_b3.shape[0], v.shape[1]))
        for n in range(v.shape[1]):
            index = 0
            for i in range(ras2):
                for j in range(i):
                    for A in range(ras3):
                        for a in range(ras2):
                            for b in range(a):
                                sig_3_out[index, n] = sig_3[A, i, j, a, b, n]
                                index = index + 1

        return np.vstack((sig_1_out, sig_2_out, sig_3_out)) + offset_v

    def _matvec(self, v):
        """Defines matrix-vector multiplication for Hamiltonian and
        guess vector.
        Input
            v -- guess vector
        Returns
            sig_out -- result of H*v multiplication
        """
        # grabbing necessary info from self
        Fa = self.Fa
        Fb = self.Fb
        tei = self.tei
        conf_space = self.conf_space
        offset = self.offset
        ras1 = self.ras1
        ras2 = self.ras2
        ras3 = self.ras3
        n_SF = self.n_SF
        delta_ec = self.delta_ec

        # offset (ROHF energy)
        offset_v = (offset * v).reshape(v.shape[0], 1)

        # do excitation schemes
        #        definitions:
        #        I      doubly occupied
        #        i,a    singly occupied
        #        A      doubly unoccupied
        v = v.reshape(v.shape[0], 1)

        # CAS EXCITATIONS
        # CAS-1SF
        if(n_SF==1 and delta_ec==0 and conf_space==""):
            return self.do_cas_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-2SF
        if(n_SF==2 and delta_ec==0 and conf_space==""):
            return self.do_cas_2sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-IP
        if(n_SF==0 and delta_ec==-1 and conf_space==""):
            return self.do_cas_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-EA
        if(n_SF==0 and delta_ec==1 and conf_space==""):
            return self.do_cas_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space==""):
            return self.do_cas_1sf_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space==""):
            return self.do_cas_1sf_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(h) EXCITATIONS
        # RAS(h)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="h"):
            return self.do_h_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-IP
        if(n_SF==0 and delta_ec==-1 and conf_space=="h"):
            return self.do_h_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-EA
        if(n_SF==0 and delta_ec==1 and conf_space=="h"):
            return self.do_h_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space=="h"):
            return self.do_h_1sf_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(p) EXCITATIONS
        # RAS(p)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="p"):
            return self.do_p_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-IP
        if(n_SF==0 and delta_ec==-1 and conf_space=="p"):
            return self.do_p_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-EA
        if(n_SF==0 and delta_ec==1 and conf_space=="p"):
            return self.do_p_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space=="p"):
            return self.do_p_1sf_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(h,p) EXCITATIONS
        # RAS(h,p)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="h,p"):
            return self.do_hp_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)



    # These two are vestigial-- I'm sure they served some purpose in the parent class,
    # but we only really need matvec for our purposes!
    def _matmat(self, v):
        """Defines matrix-vector multiplication for Hamiltonian and
        guess vector.
        Input
            v -- set of guess vectors
        Returns
            sig_out -- result of H*v multiplication
        """
        # grabbing necessary info from self
        Fa = self.Fa
        Fb = self.Fb
        tei = self.tei
        conf_space = self.conf_space
        offset = self.offset
        ras1 = self.ras1
        ras2 = self.ras2
        ras3 = self.ras3
        n_SF = self.n_SF
        delta_ec = self.delta_ec

        # offset (ROHF energy)
        offset_v = (offset * v).reshape(v.shape[0], v.shape[1])

        # do excitation schemes
        #        definitions:
        #        I      doubly occupied
        #        i,a    singly occupied
        #        A      doubly unoccupied

        # CAS EXCITATIONS
        # CAS-1SF
        if(n_SF==1 and delta_ec==0 and conf_space==""):
            return self.do_cas_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-1SF (neutral determinants only!!)
        if(n_SF==1 and delta_ec==0 and conf_space=="neutral"):
            return self.do_cas_1sf_neutral(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-2SF
        if(n_SF==2 and delta_ec==0 and conf_space==""):
            return self.do_cas_2sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-IP
        if(n_SF==0 and delta_ec==-1 and conf_space==""):
            return self.do_cas_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-EA
        if(n_SF==0 and delta_ec==1 and conf_space==""):
            return self.do_cas_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space==""):
            return self.do_cas_1sf_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # CAS-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space==""):
            return self.do_cas_1sf_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(h) EXCITATIONS
        # RAS(h)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="h"):
            return self.do_h_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-IP
        if(n_SF==0 and delta_ec==-1 and conf_space=="h"):
            return self.do_h_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-EA
        if(n_SF==0 and delta_ec==1 and conf_space=="h"):
            return self.do_h_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(h)-1SF-IP
        if(n_SF==1 and delta_ec==-1 and conf_space=="h"):
            return self.do_h_1sf_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(p) EXCITATIONS
        # RAS(p)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="p"):
            return self.do_p_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-IP
        if(n_SF==0 and delta_ec==-1 and conf_space=="p"):
            return self.do_p_ip(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-EA
        if(n_SF==0 and delta_ec==1 and conf_space=="p"):
            return self.do_p_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)
        # RAS(p)-1SF-EA
        if(n_SF==1 and delta_ec==1 and conf_space=="p"):
            return self.do_p_1sf_ea(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)

        # RAS(h,p) EXCITATIONS
        # RAS(h,p)-1SF
        if(n_SF==1 and delta_ec==0 and conf_space=="h,p"):
            return self.do_hp_1sf(v, Fa, Fb, tei, offset_v, ras1, ras2, ras3)


    def _rmatvec(self, v):
        print("rmatvec function called -- not implemented yet!!")
        return np.zeros(30)


