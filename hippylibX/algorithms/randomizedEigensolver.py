import numpy as np
from .multivector import MultiVector, MatMvMult, MvDSmatMult
from .linalg import Solver2Operator

# d, U = hpx.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)

def doublePassG(A, B, Binv, Omega, k, s=1, check=False):
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

    # has to be done later, seperately
    # if check:
    #     check_g(A,B, U, d)

    return d, U
