from __future__ import print_function
import math
import numpy as np
from numpy import linalg as LIN
from scipy.sparse import linalg as SPLIN
from linop import LinOpH
import psi4

"""
1SF-CAS PROGRAM

Runs the 1SF-CAS calculation. The (h), (p), and (h,p) excitations are still in development. 

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
# Handles full ERIs.
class ERI_Full:
    def __init__(self, eri_in):
        self.eri = eri_in

    ''' 
    Returns a given subblock of the ERI matrix.
    '''
    def get_subblock(self, a, b, c, d, blocktype=""): 
        if(blocktype==""):
            return self.eri[a[0]:a[1], b[0]:b[1], c[0]:c[1], d[0]:d[1]]
        elif(blocktype=="JK"):
            out = self.eri - self.eri.transpose(0, 1, 3, 2)
            return self.eri[a[0]:a[1], b[0]:b[1], c[0]:c[1], d[0]:d[1]]

###############################################################################################
# Functions
###############################################################################################

# Kronaker delta function.
def kdel(i, j):
    if(i==j):
        return 1
    else:
        return 0

# Gets the non-spin-adapted Fock matrix (Fa and Fb as one)
def get_nsa_F(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    # References: Psi4NumPy tutorials
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    C = np.block([[Ca, np.zeros_like(Cb)],
                      [np.zeros_like(Ca), Cb]])
    # one-electron part
    Fa = psi4.core.Matrix.to_array(wfn.Fa(), copy=True)
    Fb = psi4.core.Matrix.to_array(wfn.Fb(), copy=True)
    F = np.block([[Fa, np.zeros_like(Fa)],
                  [np.zeros_like(Fb), Fb]])
    F = np.dot(C.T, np.dot(F, C)) 
    return F

# Gets Fa and Fb for spactial orbitals
def get_spatial_F(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    C = psi4.core.Matrix.to_array(wfn.Ca())
    Fa = psi4.core.Matrix.to_array(wfn.Fa(), copy=True)
    Fb = psi4.core.Matrix.to_array(wfn.Fb(), copy=True)
    Fa = np.dot(C.T, np.dot(Fa, C)) 
    Fb = np.dot(C.T, np.dot(Fb, C)) 
    return (Fa, Fb)

# gets the non-spin-adapted two-electron integrals (alpha and beta as one)
def get_nsa_tei(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    # References: Psi4NumPy tutorials
    Ca = psi4.core.Matrix.to_array(wfn.Ca())
    Cb = psi4.core.Matrix.to_array(wfn.Cb())
    C = np.block([[Ca, np.zeros_like(Cb)],
                      [np.zeros_like(Ca), Cb]])
    # two-electron part
    mints = psi4.core.MintsHelper(wfn.basisset())
    tei = psi4.core.Matrix.to_array(mints.ao_eri(), copy=True)
    I = np.eye(2)
    tei = np.kron(I, tei)
    tei = np.kron(I, tei.T)
    tei = tei.transpose(0, 2, 1, 3) - tei.transpose(0, 2, 3, 1)
    tei = np.einsum('pqrs,pa',tei,C)
    tei = np.einsum('aqrs,qb',tei,C)
    tei = np.einsum('abrs,rc',tei,C)
    tei = np.einsum('abcs,sd',tei,C)
    return tei

# gets the spatial two-electron integrals (alpha and beta equivalent)
def get_spatial_tei(wfn):
    # get necessary integrals/matrices from Psi4 (AO basis)
    C = psi4.core.Matrix.to_array(wfn.Ca())
    # two-electron part
    mints = psi4.core.MintsHelper(wfn.basisset())
    tei = psi4.core.Matrix.to_array(mints.ao_eri(), copy=True)
    # put in physicists' notation
    tei = tei.transpose(0, 2, 1, 3)
    tei = np.einsum('pqrs,pa',tei,C)
    tei = np.einsum('aqrs,qb',tei,C)
    tei = np.einsum('abrs,rc',tei,C)
    tei = np.einsum('abcs,sd',tei,C)
    return tei 

# Performs the 1SF-CAS calculation.
# Parameters:
#    charge          The desired charge (nIP/EA determining, for later use)
#    mult            The desired multiplicity (nSF determining, for later use)
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
def do_sf_cas( charge, mult, mol, conf_space="", add_opts={}, sf_diag_method="LinOp", num_roots=6, guess_type="CAS" ):
    psi4.core.clean()
    opts = {'basis': 'cc-pvdz',
            'scf_type': 'pk',
            'e_convergence': 1e-10,
            'd_convergence': 1e-10,
            'reference': 'rohf'}
    opts.update(add_opts)
    psi4.set_options(opts)
    e, wfn = psi4.energy('scf', molecule=mol, return_wfn=True)
    Fa, Fb = get_spatial_F(wfn)
    tei = ERI_Full(get_spatial_tei(wfn))
    socc = wfn.soccpi()[0]
    na_virt = wfn.basisset().nbf() - (wfn.soccpi()[0] + wfn.doccpi()[0])
    nb_virt = wfn.basisset().nbf() - wfn.doccpi()[0]
    if(conf_space==""):
        n_dets = socc * socc
    if(conf_space=="h"):
        n_dets = (socc * socc) + (socc * wfn.doccpi()[0]) + (((socc-1)*(socc)/2) * socc * wfn.doccpi()[0])
    if(conf_space=="p"):
        n_dets = (socc * nb_virt) + (((socc-1)*(socc)/2) * socc * na_virt)
    if(conf_space=="h,p"):
        n_dets = (socc * nb_virt) + (((socc-1)*(socc)/2) * socc * na_virt) + (socc * wfn.doccpi()[0]) + (((socc-1)*(socc)/2) * socc * wfn.doccpi()[0])
    #if(sf_diag_method == "RSP"):
    #    print("FROM DIAG: ", e + np.sort(LIN.eigvalsh(H))[0:8])
    #    print("FROM DIAG: ", np.sort(LIN.eigvalsh(H))[0:6])
    #    return(e + np.sort(LIN.eigvalsh(H))[0])
    if(sf_diag_method == "LinOp"):
        #A = SPLIN.LinearOperator(H.shape, matvec=mv)
        a_occ = wfn.doccpi()[0] + wfn.soccpi()[0]
        b_occ = wfn.doccpi()[0]
        a_virt = wfn.basisset().nbf() - a_occ
        b_virt = wfn.basisset().nbf() - b_occ
        print("Number of determinants:", n_dets)
        if(conf_space==""):
            A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, conf_space_in=conf_space)
            vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots)
        else:
            if("guess_type"=="CAS"):
                cas_A = LinOpH((socc*socc,socc*socc), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, conf_space_in="")
                cas_vals, cas_vects = SPLIN.eigsh(cas_A, which='SA', k=num_roots)
                socc = wfn.soccpi()[0]
                v3_guess = np.zeros((n_dets-(socc*socc), num_roots))
                guess_vect = np.vstack((cas_vects, v3_guess)).T
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, k=num_roots, which='SA', v0=guess_vect[0, :])
            else:
                A = LinOpH((n_dets,n_dets), a_occ, b_occ, a_virt, b_virt, Fa, Fb, tei, conf_space_in=conf_space)
                vals, vects = SPLIN.eigsh(A, which='SA', k=num_roots)
        for i, corr in enumerate(vals):
            print("ROOT %i: %6.6f" % (i, e + corr))
        return(e + vals[0])
