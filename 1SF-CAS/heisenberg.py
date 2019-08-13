import numpy as np
import numpy.linalg as LIN

class heis_ham:

    def __init__(self):
        sig_x = 0.5*np.array([[0.0, 1.0],[1.0, 0.0]])
        sig_y = 0.5*np.array([[0.0, 0.0-1.0j],[0.0+1.0j, 0.0]])
        sig_z = 0.5*np.array([[1.0, 0.0],[0.0, -1.0]])
        self.pauli = {'x': sig_x, 'y': sig_y, 'z': sig_z}
        self.H = None
        self.H_diag = None
        self.Sz_diag = None

    '''
    Does many Kronaker products... Recursively.
    non_I -- the paired indexes (i,j) that aren't the identity
    p -- which pauli to use (x/y/z)
    index -- current index/depth
    max_index -- depth at which to end recursion
    '''
    def recursive_kron(self, non_I, p, index, max_index):
        if(index in non_I):
            tmp = self.pauli[p]
        else:
            tmp = np.array([[1.0, 0.0],[0.0, 1.0]])
        if(index == max_index):
            return tmp
        else:
            return np.kron(tmp, self.recursive_kron(non_I, p, index+1, max_index))

    def do_heisenberg(self, n_sites, J):
        max_ind = n_sites - 1
        H = np.zeros((2**n_sites, 2**n_sites))
        Sz = np.zeros((2**n_sites, 2**n_sites))
        for i in range(n_sites):
            Sz = Sz + self.recursive_kron([i], 'z', 0, max_ind)
            for j in range(i):
                H = H - 2.0*J[i,j]*(self.recursive_kron([i,j], 'x', 0, max_ind) +
                                self.recursive_kron([i,j], 'y', 0, max_ind) +
                                self.recursive_kron([i,j], 'z', 0, max_ind) )
        c = 0.3891124
        tmp, HS_vecs = LIN.eigh(H + c*Sz)
        self.H = H
        self.H_diag = np.real(np.dot(HS_vecs.T, np.dot(H, HS_vecs)))
        self.Sz_diag = np.real(np.dot(HS_vecs.T, np.dot(Sz, HS_vecs)))

    def print_H(self):
        print("\nHeisenberg Hamiltonian:")
        print(self.H)

    def print_roots(self):
        print("\nHeisenberg Hamiltonian Roots:")
        print("\tEnergy\t\tSz")
        for i in range(self.H_diag.shape[0]):
            print("\t%12.12f\t%6.6f" %(self.H_diag[i,i], self.Sz_diag[i,i]))
        print()




