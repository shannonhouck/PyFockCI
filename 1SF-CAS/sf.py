from __future__ import print_function
import math
import numpy as np
from numpy import linalg as LIN
from scipy.sparse import linalg as SPLIN
from linop import LinOpH
import psi4

"""
1SF-CAS PROGRAM

Runs the 1SF-CAS calculation. Currently implementing 1SF-IP/EA methods.

Refs:
Crawford Tutorials (http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project12)
DePrince Tutorials (https://www.chem.fsu.edu/~deprince/programming_projects/cis/)
Sherrill Notes (http://vergil.chemistry.gatech.edu/notes/cis/cis.html)
Psi4NumPy Tutorials
"""

###############################################################################################
# Classes
###############################################################################################

# Class for two-electron integral object handling.
# Handles ERIs.
class ERI:
    def __init__(self, wfn):
        pass

    ''' 
    Returns a given subblock of the ERI matrix.
    '''
    def get_subblock(self, a, b, c, d): 
        pass

# Class for full ERI integrals.
class ERI_Full(ERI):
    def __init__(self, C, basis, ras1, ras2, ras3):
        # get necessary integrals/matrices from Psi4 (AO basis)
        mints = psi4.core.MintsHelper(basis)
        self.eri = psi4.core.Matrix.to_array(mints.ao_eri())
        # put in physicists' notation
        self.eri = self.eri.transpose(0, 2, 1, 3)
        # move to MO basis
        self.eri = np.einsum('pqrs,pa',self.eri,C)
        self.eri = np.einsum('aqrs,qb',self.eri,C)
        self.eri = np.einsum('abrs,rc',self.eri,C)
        self.eri = np.einsum('abcs,sd',self.eri,C)
        # ind stores the indexing of ras1/ras2/ras3 for get_subblock method
        self.ind = [[0,0],[0,ras1],[ras1,ras1+ras2],[ras1+ras2,ras1+ras2+ras3]]

    def get_subblock(self, a, b, c, d):
        return self.eri[self.ind[a][0]:self.ind[a][1], self.ind[b][0]:self.ind[b][1], self.ind[c][0]:self.ind[c][1], self.ind[d][0]:self.ind[d][1]]

# Class for full ERI integrals.
class ERI_DF(ERI):
    # Used Psi4NumPy for reference for this section
    # ras1, ras2, ras3 indicate the number of orbitals in each section
    def __init__(self, C, basis, aux, ras1, ras2, ras3, conf_space):
        # get info from Psi4
        zero = psi4.core.BasisSet.zero_ao_basis_set()
        mints = psi4.core.MintsHelper(basis)
        # (Q|pq)
        eri = psi4.core.Matrix.to_array(mints.ao_eri(zero, aux, basis, basis))
        eri = np.squeeze(eri)
        # J^-1/2 (don't need to keep)
        #J = psi4.core.Matrix.to_array(mints.ao_eri(zero, aux, zero, aux))
        J = mints.ao_eri(zero, aux, zero, aux)
        J.power(-0.5, 1e-14)
        J = np.squeeze(J)
        # Contract and obtain final form
        eri = np.einsum("PQ,Qpq->Ppq", J, eri)
        # put in physicists' notation
        #self.eri = self.eri.transpose(0, 2, 1, 3)
        C = psi4.core.Matrix.to_array(C)
        C_ras1 = C[:, 0:ras1]
        C_ras2 = C[:, ras1:ras1+ras2]
        C_ras3 = C[:, ras1+ras2:]
        # move to MO basis
        # Notation: ij in active space, IJ in docc, AB in virtual
        # Bnm notation, where n and m indicate RAS space (1/2/3)
        # all of them need RAS2
        B2m = np.einsum('Ppq,pi->Piq', eri, C_ras2)
        self.B22 = np.einsum('Piq,qj->Pij', B2m, C_ras2)
        # if configuration space is "h"
        if(conf_space == "h"):
            B1m = np.einsum('Ppq,pI->PIq', eri, C_ras1)
            self.B11 = np.einsum('PIq,qJ->PIJ', B1m, C_ras1)
            self.B12 = np.einsum('PIq,qj->PIj', B1m, C_ras2)
            self.B21 = np.einsum('Piq,qJ->PiJ', B2m, C_ras1)
        if(conf_space == "p"):
            B3m = np.einsum('Ppq,pA->PAq', eri, C_ras3)
            self.B33 = np.einsum('PAq,qB->PAB', B3m, C_ras3)
            self.B32 = np.einsum('PAq,qj->PAj', B3m, C_ras2)
            self.B23 = np.einsum('Piq,qA->PiA', B2m, C_ras3)
        if(conf_space == "h,p"):
            B1m = np.einsum('Ppq,pI->PIq', eri, C_ras1)
            self.B11 = np.einsum('PIq,qJ->PIJ', B1m, C_ras1)
            self.B12 = np.einsum('PIq,qj->PIj', B1m, C_ras2)
            self.B21 = np.einsum('Piq,qJ->PiJ', B2m, C_ras1)
            self.B13 = np.einsum('PIq,qA->PIA', B1m, C_ras3)
            B3m = np.einsum('Ppq,pA->PAq', eri, C_ras3)
            self.B33 = np.einsum('PAq,qB->PAB', B3m, C_ras3)
            self.B32 = np.einsum('PAq,qj->PAj', B3m, C_ras2)
            self.B31 = np.einsum('PAq,qJ->PAJ', B3m, C_ras1)
            self.B23 = np.einsum('Piq,qA->PiA', B2m, C_ras3)

    # a, b, c, d defined as RAS(1/2/3) spaces, integers
    def get_subblock(self, a, b, c, d):
        # the B matrices are still in chemists' notation, don't switch b/c until end
        if(a == 1):
            if(c == 1):
                B_bra = self.B11
            if(c == 2):
                B_bra = self.B12
            if(c == 3):
                B_bra = self.B13
        if(a == 2):
            if(c == 1):
                B_bra = self.B21
            if(c == 2):
                B_bra = self.B22
            if(c == 3):
                B_bra = self.B23
        if(a == 3):
            if(c == 1):
                B_bra = self.B31
            if(c == 2):
                B_bra = self.B32
            if(c == 3):
                B_bra = self.B33

        if(b == 1):
            if(d == 1):
                B_ket = self.B11
            if(d == 2):
                B_ket = self.B12
            if(d == 3):
                B_ket = self.B13
        if(b == 2):
            if(d == 1):
                B_ket = self.B21
            if(d == 2):
                B_ket = self.B22
            if(d == 3):
                B_ket = self.B23
        if(b == 3):
            if(d == 1):
                B_ket = self.B31
            if(d == 2):
                B_ket = self.B32
            if(d == 3): 
                B_ket = self.B33

        return np.einsum("Pij,Pkl->ijkl", B_bra, B_ket).transpose(0, 2, 1, 3)

###############################################################################################
# Functions
###############################################################################################

# Kronaker delta function.
def kdel(i, j):
    if(i==j):
        return 1
    else:
        return 0

# Gets Fa and Fb for spactial orbitals
def get_F(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    C = psi4.core.Matrix.to_array(wfn.Ca())
    Fa = psi4.core.Matrix.to_array(wfn.Fa(), copy=True)
    Fb = psi4.core.Matrix.to_array(wfn.Fb(), copy=True)
    Fa = np.dot(C.T, np.dot(Fa, C)) 
    Fb = np.dot(C.T, np.dot(Fb, C)) 
    return (Fa, Fb)

# calculates S**2
def calc_s_squared(n_SF, delta_ec, conf_space, vect, socc):
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        s2_vect = np.zeros(vect.shape)
        count = 0
        for i in range(socc):
            for a in range(socc):
                if(i==a): # to same orbital, no mult lost
                    s = (socc)/2.0
                else: # to different orbital, S-1
                    s = (socc - 1.0)/2.0
                s2_vect[count] = vect[count]*(s*(s+1.0))
                count = count+1
        return np.einsum("i,i->", vect, s2_vect)
    else:
        print("S**2 value for %iSF with electron count change of %i not yet supported." %(n_SF, delta_ec) )
        return 0

# Performs the 1SF-CAS calculation.
# Parameters:
#    delta_a         Desired number of alpha electrons to remove
#    delta_b         Desired number of beta electrons to add
#    mol             Molecule to run calculation on
#    conf_space      Desired excitation scheme:
#                        ""       1SF-CAS
#                        "h"      1SF-CAS + h
#                        "p"      1SF-CAS + p
#                        "h,p"    1SF-CAS + (h,p)
#    add_opts        Additional options (to be passed on to Psi4)
#    sf_diag_method  Diagonalization method to use.
#                        "RSP"    Direct
#                        "LinOp"  LinOp
#    num_roots       Number of roots to solve for.
#    guess_type      Type of guess vector to use (CAS vs. RANDOM)
#
# Returns:
#    energy          Lowest root found by eigensolver (energy of system)
def do_sf_cas( delta_a, delta_b, mol, conf_space="", add_opts={}, sf_diag_method="LinOp", num_roots=6, guess_type="CAS", integral_type="FULL", aux_basis_name="", e_convergence=1e-10 ):
    # cleanup in case of multiple calculations
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    # setting default options, reading in additional options from user
    opts = {'basis': 'cc-pvdz',
            'scf_type': 'direct',
            'e_convergence': 1e-10,
            'd_convergence': 1e-10,
            'reference': 'rohf'}
    opts.update(add_opts)
    psi4.set_options(opts)
    # run ROHF
    e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    # obtain some important values
    socc = wfn.soccpi()[0]
    na_virt = wfn.basisset().nbf() - (wfn.soccpi()[0] + wfn.doccpi()[0])
    nb_virt = wfn.basisset().nbf() - wfn.doccpi()[0]
    # get integrals
    Fa, Fb = get_F(wfn)
    if(integral_type=="FULL"):
        tei = ERI_Full(wfn.Ca(), wfn.basisset(), wfn.doccpi()[0], socc, na_virt)
    if(integral_type=="DF"):
        # if user hasn't defined which aux basis to use, default behavior is to use the one from opts
        if(aux_basis_name == ""):
            aux_basis_name = opts['basis']
        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", aux_basis_name)
        tei = ERI_DF(wfn.Ca(), wfn.basisset(), aux_basis, wfn.doccpi()[0], socc, na_virt, conf_space)
    # determine number of spin-flips and total change in electron count
    delta_ec = delta_b - delta_a
    n_SF = min(delta_a, delta_b)
    # determine number of determinants
    # RAS-1SF
    if(n_SF==1 and delta_ec==0 and conf_space==""):
        n_dets = socc * socc
    elif(n_SF==2 and delta_ec==0 and conf_space==""):
        guess_type = ""
        n_dets = 0
        for i in range(socc):
            for j in range(i):
                for a in range(socc):
                    for b in range(a):
                        n_dets = n_dets + 1 
    elif(n_SF==1 and delta_ec==0 and conf_space=="h"):
        n_dets = (socc * socc) + (socc * wfn.doccpi()[0]) + (((socc-1)*(socc)/2) * socc * wfn.doccpi()[0])
    elif(n_SF==1 and delta_ec==0 and conf_space=="p"):
        n_dets = (socc * nb_virt) + (((socc-1)*(socc)/2) * socc * na_virt)
    elif(n_SF==1 and delta_ec==0 and (conf_space=="h,p" or conf_space=="1x")):
        n_dets = (socc * nb_virt) + (((socc-1)*(socc)/2) * socc * na_virt) + (socc * wfn.doccpi()[0]) + (((socc-1)*(socc)/2) * socc * wfn.doccpi()[0])
    # CAS-IP/EA
    elif(n_SF==0 and (delta_ec==-1 or delta_ec==1) and conf_space==""):
        guess_type = ""
        n_dets = socc
    # RAS(h)-EA
    elif(n_SF==0 and delta_ec==1 and conf_space=="h"):
        guess_type = ""
        n_dets = socc
        for I in range(wfn.doccpi()[0]):
            for a in range(socc):
                for b in range(a):
                    n_dets = n_dets + 1
    # RAS(p)-EA
    elif(n_SF==0 and delta_ec==1 and conf_space=="p"):
        guess_type = ""
        n_dets = socc + na_virt + (na_virt*socc*socc)
    # RAS(h)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="h"):
        guess_type = ""
        n_dets = socc + wfn.doccpi()[0] + (wfn.doccpi()[0]*socc*socc)
    # RAS(p)-IP
    elif(n_SF==0 and delta_ec==-1 and conf_space=="p"):
        guess_type = ""
        n_dets = socc
        for i in range(socc):
            for j in range(i):
                for A in range(na_virt):
                    n_dets = n_dets + 1
    # CAS-1SF-IP/EA
    elif(n_SF==1 and (delta_ec==-1 or delta_ec==1) and conf_space==""):
        guess_type = ""
        n_dets = socc * ((socc-1)*(socc)/2)
    # RAS(p)-1SF-EA
    elif(n_SF==1 and delta_ec==1 and conf_space=="p"):
            n_dets = socc * na_virt * socc
            for i in range(socc):
                for a in range(socc):
                    for b in range(a):
                        n_dets = n_dets + 1
            for i in range(socc):
                for j in range(i):
                    for A in range(na_virt):
                        for a in range(socc):
                            for b in range(a):
                                n_dets = n_dets + 1
    # RAS(h)-1SF-IP
    elif(n_SF==1 and delta_ec==-1 and conf_space=="h"):
        guess_type = ""
        nb_occ = wfn.doccpi()[0]
        n_dets = socc * ((socc-1)*(socc)/2)
        n_dets = n_dets + (socc * nb_occ * socc)
        # this is the MOST hack-ish way to get the triangle number of a traingle number. Fix later
        count = 0 
        for I in range(nb_occ):
            for i in range(socc):
                for j in range(i):
                    for a in range(socc):
                        for b in range(a):
                            count = count + 1
        n_dets = n_dets + count
    else:
        print("Sorry, %iSF with electron count change of %i not yet supported. Exiting..." %(n_SF, delta_ec) )
        exit()
    #if(sf_diag_method == "RSP"):
    #    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0:8])
    #    print("FROM DIAG: ", np.sort(LIN.eigvalsh(H))[0:6])
    #    return(e + np.sort(LIN.eigvalsh(H))[0])
    print("Performing %iSF with electron count change of %i..." %(n_SF, delta_ec) )
    if(sf_diag_method == "LinOp"):
        #A = SPLIN.LinearOperator(H.shape, matvec=mv)
        a_occ = wfn.doccpi()[0] + wfn.soccpi()[0]
        b_occ = wfn.doccpi()[0]
        a_virt = wfn.basisset().nbf() - a_occ
        b_virt = wfn.basisset().nbf() - b_occ
        print("Number of determinants:", n_dets)
        if( num_roots >= n_dets ):
            num_roots = n_dets - 1
        if(conf_space==""):
            A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, n_SF, delta_ec, conf_space_in=conf_space)
            vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots, tol=e_convergence)
        else:
            if("guess_type"=="CAS"):
                cas_A = LinOpH((socc*socc,socc*socc), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, n_SF, delta_ec, conf_space_in="")
                cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA', k=1, tol=e_convergence)
                socc = wfn.soccpi()[0]
                v3_guess = np.zeros((n_dets-(socc*socc), 1))
                guess_vect = np.vstack((cas_vects, v3_guess)).T
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, n_SF, delta_ec, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA', v0=guess_vect, tol=e_convergence)
            else:
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, n_SF, delta_ec, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots, tol=e_convergence)
        print("\nROOT No.\tEnergy\t\tS**2")
        print("------------------------------------------------")
        for i, corr in enumerate(vals):
            print("   %i\t\t%6.6f\t%8.6f" % (i, e + corr, calc_s_squared(n_SF, delta_ec, conf_space, vects[:, i], socc)))
        print("------------------------------------------------\n")
        return(e + vals[0])

