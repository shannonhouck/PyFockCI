import sf_ip_ea
from sf_ip_ea import solvers
import numpy as np
import scipy.linalg as LIN
import pytest

# Test Davidson

class DavTestMat:
    def __init__(self, dim, offset):
        self.A = np.random.rand(dim, dim)
        self.A = self.A + self.A.T
        for i in range(dim):
            self.A[i,i] += offset
        self.shape = self.A.shape

    def matvec(self, v):
        return np.dot(self.A, v)

    def matmat(self, v):
        return np.dot(self.A, v)

    def diag(self):
        return np.diag(self.A)


@pytest.mark.davidsontest
def test_1():
    n_roots = 1000
    dim = 1000
    offset = 0
    A = DavTestMat(dim, offset)
    guess = LIN.orth(np.random.rand(dim, n_roots))
    evals, evecs = solvers.davidson(A, guess)
    evals_ref = np.linalg.eigvalsh(A.A)
    assert np.allclose(evals, evals_ref)

@pytest.mark.davidsontest
def test_2():
    n_roots = 1000
    dim = 1000
    offset = -5000
    A = DavTestMat(dim, offset)
    guess = LIN.orth(np.random.rand(dim, n_roots))
    evals, evecs = solvers.davidson(A, guess)
    evals_ref = np.linalg.eigvalsh(A.A)
    assert np.allclose(evals, evals_ref)

