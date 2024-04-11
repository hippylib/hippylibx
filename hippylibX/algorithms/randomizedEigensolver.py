import numpy as np
import petsc4py.PETSc
from .multivector import MultiVector, MatMvMult, MvDSmatMult
from .linalg import Solver2Operator
import petsc4py
from typing import Any


def doublePassG(
    A: petsc4py.PETSc.Mat,
    B: petsc4py.PETSc.Mat,
    Binv: Any,
    Omega: MultiVector,
    k: int,
    s=1,
) -> tuple[np.array, MultiVector]:
    nvec = Omega.nvec
    assert nvec >= k
    Ybar = MultiVector.createFromVec(Omega[0], nvec)
    Q = MultiVector.createFromMultiVec(Omega)
    # Bringing the orthogonalization inside of the power iteration could improve accuracy
    for i in range(s):
        MatMvMult(A, Q, Ybar)
        MatMvMult(Solver2Operator(Binv), Ybar, Q)  # noqa

    Q.Borthogonalize(B)
    AQ = MultiVector.createFromVec(Omega[0], nvec)
    MatMvMult(A, Q, AQ)

    T = AQ.dot(Q)

    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()

    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]]

    U = MultiVector.createFromVec(Omega[0], k)
    MvDSmatMult(Q, V, U)

    return d, U
