import numpy as np
import petsc4py.PETSc
from .multivector import MultiVector, MatMvMult, MvDSmatMult
from .linalg import Solver2Operator
import petsc4py
from typing import Union
from ..modeling.prior import _BilaplacianRsolver
# d, U = hpx.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)

def doublePassG(A : petsc4py.PETSc.Mat, B : petsc4py.PETSc.Mat, Binv : Union[_BilaplacianRsolver,petsc4py.PETSc.KSP], Omega : MultiVector, k, s=1, check=False):
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

    if check:
        check_g(A,B, U, d)

    return d, U

def check_g(A : petsc4py.PETSc.Mat, B : petsc4py.PETSc.Mat, U : MultiVector , d : np.array):
    nvec  = U.nvec
    AU = MultiVector.createFromVec(U[0], nvec)
    BU = MultiVector.createFromVec(U[0], nvec)
    MatMvMult(A, U, AU)
    MatMvMult(B, U, BU)
    
    # Residual checks
    diff = MultiVector.createFromMultiVec(AU)
    diff.axpy(-d, BU)

    # res_norms = diff.norm("l2")
    res_norms = diff.norm(petsc4py.PETSc.NormType.N2)
    
    # B-ortho check
    UtBU = BU.dot(U)
    
    err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')
    
    #A-ortho check
    V = MultiVector.createFromMultiVec(U)
    scaling = np.power(np.abs(d), -0.5)
    V.scale(scaling)
    AU.scale(scaling)
    VtAV = AU.dot(V)
    err = VtAV - np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, 'fro')
    
    rank = A.getComm().rank

    if rank == 0:
        print( "|| UtBU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambdaBu||_2")
        for i in range(res_norms.shape[0]):
            print( "{0:5e} {1:5e}".format(d[i], res_norms[i]) ) 