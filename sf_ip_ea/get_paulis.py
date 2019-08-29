import sys
import pickle
import heisenberg
import numpy as np
import numpy.linalg as LIN


pauli = []
pauli.append({})

def gen_pauli(s):
    heis = heisenberg.heis_ham(use_pickle=False)
    dim = heis.recursive_kron([], 1, 'x', 0, s-1).shape[0]
    J = np.ones((dim,dim)) - np.eye(dim)
    heis.do_heisenberg(s*[1], J)

    np.set_printoptions(linewidth=1000)

    # accounts for phase
    #if(heis.Sz_diag[0,0] < 0):
    #    p = -1.0
    #else:
    #p = 1.0

    sig_s2 = heis.S2_diag[:s+1, :s+1]

    # reorder eigenvectors
    vecs = heis.vecs[:, :s+1]
    Sz = heis.Sz_diag
    ind = np.argsort(np.diagonal(Sz[:s+1, :s+1]))[::-1]
    vecs = vecs[:, ind]

    Sz = heis.Sz
    Sx = heis.Sx
    Sy = heis.Sy
    sig_x = np.real(np.dot(vecs.T, np.dot(Sx, vecs)))
    sig_y = np.dot(vecs.T, np.dot(Sy, vecs))
    sig_z = np.real(np.dot(vecs.T, np.dot(Sz, vecs)))

    if(sig_x[1,0] < 0):
        sig_x = -1.0*sig_x
    if(np.imag(sig_y[1,0]) < 0):
        sig_y = -1.0*sig_y

    print("Sx")
    print(sig_x)
    print("Sy")
    print(sig_y)
    print("Sz")
    print(sig_z)

    I = np.eye(sig_x.shape[0])
    return {'x': sig_x, 'y': sig_y, 'z': sig_z, 's2': sig_s2, 'I': I}

for s in range(1, 6):
    print("S=", s, "/2")
    pauli.append(gen_pauli(s))

print(pauli)

with open('paulis.pickle', 'wb') as f:
    pickle.dump(pauli, f, pickle.HIGHEST_PROTOCOL)


