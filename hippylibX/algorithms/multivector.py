import numpy as np
import petsc4py
import copy


class MultiVector:
    def __init__(self, example_vec=None, nvec=None, mv=None):
        if mv is None:  # original
            self.data = []
            self.nvec = nvec

            for i in range(self.nvec):
                self.data.append(example_vec.duplicate())
        else:  # copy
            self.nvec = mv.nvec
            self.data = copy.deepcopy(mv.data)

    def __del__(self):
        for d in self.data:
            d.destroy()

    def __getitem__(self, k):
        return self.data[k]

    def scale(self, alpha):
        if isinstance(alpha, float):
            for d in self.data:
                d.scale(alpha)
        elif isinstance(alpha, np.array):
            for i, d in enumerate(self.data):
                d.scale(alpha[i])

    def dot(self, v) -> np.array:
        if isinstance(v, petsc4py.PETSc.Vec):
            return_values = []
            for i in range(self.nvec):
                return_values.append(self[i].dot(v))
            return np.array(return_values)

        elif isinstance(v, MultiVector):
            return_values = []
            for i in range(self.nvec):
                return_row = []
                for j in range(v.nvec):
                    return_row.append(self[i].dot(v[j]))
                return_values.append(return_row)
            return np.array(return_values)

    def reduce(self, alpha: np.array) -> petsc4py.PETSc.Vec:
        return_vec = self[0].duplicate()
        return_vec.scale(0.0)
        for i in range(self.nvec):
            return_vec.axpy(alpha[i], self[i])
        return return_vec

    def Borthogonalize(self, B):
        return self._mgs_stable(B)

    def _mgs_stable(self, B: petsc4py.PETSc.Mat):
        n = self.nvec
        Bq = MultiVector(self[0], n)
        r = np.zeros((n, n), dtype="d")
        reorth = np.zeros((n,), dtype="d")
        eps = np.finfo(np.float64).eps

        for k in np.arange(n):
            B.mult(self[k], Bq[k])
            t = np.sqrt(Bq[k].dot(self[k]))

            nach = 1
            u = 0
            while nach:
                u += 1
                for i in np.arange(k):
                    s = Bq[i].dot(self[k])
                    r[i, k] += s
                    self[k].axpy(-s, self[i])

                B.mult(self[k], Bq[k])
                tt = np.sqrt(Bq[k].dot(self[k]))
                if tt > t * 10.0 * eps and tt < t / 10.0:
                    nach = 1
                    t = tt
                else:
                    nach = 0
                    if tt < 10.0 * eps * t:
                        tt = 0.0

            reorth[k] = u
            r[k, k] = tt
            if np.abs(tt * eps) > 0.0:
                tt = 1.0 / tt
            else:
                tt = 0.0

            self[k].scale(tt)
            Bq[k].scale(tt)

        return Bq, r


def MatMvMult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    if hasattr(A, "matMvMult"):
        A.matMvMult(x, y)
    else:
        for i in range(x.nvec()):
            A.mult(x[i], y[i])


def MatMvTranspmult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    assert hasattr(A, "transpmult"), "A does not have transpmult method implemented"
    if hasattr(A, "matMvTranspmult"):
        A.matMvTranspmult(x, y)
    else:
        for i in range(x.nvec()):
            A.multTranspose(x[i], y[i])


def MvDSmatMult(X, A, Y):
    assert (
        X.nvec() == A.shape[0]
    ), "X Number of vecs incompatible with number of rows in A"
    assert (
        Y.nvec() == A.shape[1]
    ), "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        Y[j].zero()
        X.reduce(Y[j], A[:, j].flatten())