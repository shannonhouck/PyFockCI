import numpy as np
import pickle
import os
import math
import numpy.linalg as LIN

class heis_ham:

    def __init__(self, use_pickle=True):
        # if we have pre-generated Pauli matrices, use those
        if(use_pickle and os.path.isfile('./paulis.pickle')):
            with open('paulis.pickle', 'rb') as f:
                self.pauli = pickle.load(f)
        # otherwise, use hardcoded S=1 case
        # for other cases you MUST generate a pauli.pickle dictionary first!!
        else:
            self.pauli = []
            self.pauli.append({})
            sig_x_1 = 0.5*np.array([[0.0, 1.0],
                                    [1.0, 0.0]])
            sig_y_1 = 0.5*np.array([[0.0, 0.0-1.0j],
                                    [0.0+1.0j, 0.0]])
            sig_z_1 = 0.5*np.array([[1.0, 0.0],
                                    [0.0, -1.0]])
            sig_s2 = np.dot(sig_x_1, sig_x_1) + np.dot(sig_y_1, sig_y_1) + np.dot(sig_z_1, sig_z_1)
            I_1 = np.eye(2)
            self.pauli.append({'x': sig_x_1, 'y': sig_y_1, 'z': sig_z_1, 's2': sig_s2, 'I': I_1})
        # store matrices generated for later access
        self.vecs = None
        self.H = None
        self.H_diag = None
        self.Sx = None
        self.Sy = None
        self.Sz = None
        self.Sz_diag = None
        self.S2 = None
        self.S2_diag = None

    '''
    Does many Kronaker products... Recursively.
    non_I -- the paired indexes (i,j) that aren't the identity
    p -- which pauli to use (x/y/z)
    index -- current index/depth
    max_index -- depth at which to end recursion
    '''
    def recursive_kron(self, non_I, s, p, index, max_index):
        if(index in non_I):
            tmp = self.pauli[s][p]
        else:
            tmp = self.pauli[s]['I']
        if(index == max_index):
            return tmp
        else:
            return np.kron(tmp, self.recursive_kron(non_I, s, p, index+1, max_index))

    def do_heisenberg(self, sites, J):
        """Does Heisenberg Hamiltonian.
              sites -- the number of electrons on each site (list).
              J -- NumPy array of J couplings between sites
        """
        n_sites = len(sites)
        max_ind = n_sites - 1
        n_dets = 1
        for s in sites:
            n_dets = n_dets * (s+1.0)
        n_dets = int(n_dets)
        H = np.zeros((n_dets,n_dets))
        Sx = np.zeros((n_dets,n_dets))
        Sy = np.zeros((n_dets,n_dets))
        Sz = np.zeros((n_dets,n_dets))
        S2 = np.zeros((n_dets,n_dets))
        for i, s in enumerate(sites):
            Sz = Sz + self.recursive_kron([i], s, 'z', 0, max_ind)
            Sx = Sx + self.recursive_kron([i], s, 'x', 0, max_ind)
            Sy = Sy + self.recursive_kron([i], s, 'y', 0, max_ind)
            S2 = S2 + self.recursive_kron([i,i], s, 's2', 0, max_ind)
            for j in range(i):
                S2 = S2 + 2.0*(self.recursive_kron([i,j], s, 'x', 0, max_ind) +
                                self.recursive_kron([i,j], s, 'y', 0, max_ind) +
                                self.recursive_kron([i,j], s, 'z', 0, max_ind) )
                H = H - 2.0*J[i,j]*(self.recursive_kron([i,j], s, 'x', 0, max_ind) +
                                self.recursive_kron([i,j], s, 'y', 0, max_ind) +
                                self.recursive_kron([i,j], s, 'z', 0, max_ind) )
        c1 = 0.3891124
        tmp, HS_vecs = LIN.eigh(H + c1*Sz)
        self.H = H
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz
        self.S2 = S2
        self.vecs = HS_vecs
        self.H_diag = np.real(np.dot(HS_vecs.T, np.dot(H, HS_vecs)))
        self.Sz_diag = np.real(np.dot(HS_vecs.T, np.dot(Sz, HS_vecs)))
        self.S2_diag = np.real(np.dot(HS_vecs.T, np.dot(S2, HS_vecs)))

    def print_H(self):
        print("\nHeisenberg Hamiltonian:")
        print(self.H)

    def print_roots(self):
        print("\nHeisenberg Hamiltonian Roots:")
        print("\tEnergy\t\tS**2")
        for i in np.argsort(np.diagonal(self.H_diag)):
            print("\t%12.12f\t%6.6f" %(self.H_diag[i,i], self.S2_diag[i,i]))
        print()




