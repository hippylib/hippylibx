# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import numpy as np
import petsc4py
from typing import Union
from typing import Type
import petsc4py.PETSc


class MultiVector:
    """
    Multivector object is created as a list of PETSc.petsc4py.Vec objects.
    """

    def __init__(self, example_vec: petsc4py.PETSc.Vec, nvec: int):
        self.nvec = nvec
        self.data = []
        for i in range(self.nvec):
            self.data.append(example_vec.duplicate())

    @classmethod
    def createFromVec(
        cls, example_vec: petsc4py.PETSc.Vec, nvec: int
    ) -> Type["MultiVector"]:
        """
        Create multivector from sample petsc4py vector whose parallel distribution is to be replicated.
        """
        return cls(example_vec, nvec)

    @classmethod
    def createFromMultiVec(cls, mv: Type["MultiVector"]) -> Type["MultiVector"]:
        """
        Create multivector from another MultiVector whose parallel distribution is to be replicated.
        """
        mv_copy = cls(mv[0], mv.nvec)
        for i in range(mv_copy.nvec):
            mv_copy.data[i] = mv.data[i].duplicate(mv.data[i].getArray())

        return mv_copy

    def __del__(self) -> None:
        for d in self.data:
            d.destroy()

    def __getitem__(self, k: int) -> petsc4py.PETSc.Vec:
        return self.data[k]

    def scale(self, alpha: Union[float, np.ndarray]) -> None:
        """
        Scale each value in the Multivector - either by a single float value or a numpy array of float values.
        """
        if isinstance(alpha, float):
            for d in self.data:
                d.scale(alpha)
        else:
            for i, d in enumerate(self.data):
                d.scale(alpha[i])

    def dot(self, v: Union[petsc4py.PETSc.Vec, Type["MultiVector"]]) -> np.array:
        """
        Perform dot product of a MultiVector object and petsc4py Vec object, store result in numpy array.
        """
        if isinstance(v, petsc4py.PETSc.Vec):
            return_values = np.zeros(self.nvec)
            for i in range(self.nvec):
                return_values[i] = self[i].dot(v)

        elif isinstance(v, MultiVector):
            return_values = np.zeros((self.nvec, v.nvec))
            for i in range(self.nvec):
                for j in range(v.nvec):
                    return_values[i, j] = self[i].dot(v[j])

        return return_values

    def reduce(self, y: petsc4py.PETSc.Vec, alpha: np.array) -> None:
        """
        Reduction of petsc4py Vec using values in a numpy array stored in each data element of MultiVector object.
        """
        for i in range(self.nvec):
            y.axpy(alpha[i], self[i])

    def axpy(self, alpha: Union[float, np.array], Y: Type["MultiVector"]) -> None:
        """
        Reduction of MultiVector object with a float or values in a numpy array stored in another MultiVector object.
        """
        if isinstance(alpha, float):
            for i in range(self.nvec):
                self[i].axpy(alpha, Y[i])
        else:
            for i in range(self.nvec):
                self[i].axpy(alpha[i], Y[i])

    def norm(self, norm_type: petsc4py.PETSc.NormType) -> np.array:
        """
        Return numpy array containing norm of each data element of MultiVector.
        """
        norm_vals = np.zeros(self.nvec)
        for i in range(self.nvec):
            norm_vals[i] = self[i].norm(norm_type)
        return norm_vals

    def Borthogonalize(self, B: petsc4py.PETSc.Mat):
        """
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic

        .. math::
            QR \\,= \\,Z, && (1),

            Q^*BQ\\, = \\, I, && (2),

            Q^*BZ \\, = \\,R, && (3),

            ZR^{-1} \\, = \\, Q, && (4).

        Returns:

            :code:`Bq` of type :code:`MultiVector` -> :code:`B`:math:`^{-1}`-orthogonal vectors
            :code:`r` of type :code:`ndarray` -> The :math:`r` of the QR decomposition.

        .. note:: :code:`self` is overwritten by :math:`Q`.
        """

        return self._mgs_stable(B)

    def _mgs_stable(
        self, B: petsc4py.PETSc.Mat
    ) -> tuple[Type["MultiVector"], np.array]:
        """ 
        Returns :math:`QR` decomposition of self, which satisfies conditions (1)--(4).
        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the :math:`B`-orthogonal :math:`QR` factorization.
        
        References:
            1. `A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized \
            Hermitian Eigenvalue Problems with application to computing \
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885`
            2. `W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980`
        
        https://github.com/arvindks/kle
        
        """

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


def MatMvMult(A: petsc4py.PETSc.Mat, x: MultiVector, y: MultiVector) -> None:
    assert x.nvec == y.nvec, "x and y have non-matching number of vectors"
    if hasattr(A, "matMvMult"):
        A.matMvMult(x, y)
    else:
        for i in range(x.nvec):
            A.mult(x[i], y[i])


def MatMvTranspmult(A: petsc4py.PETSc.Mat, x: MultiVector, y: MultiVector) -> None:
    assert x.nvec == y.nvec, "x and y have non-matching number of vectors"
    assert hasattr(A, "transpmult"), "A does not have transpmult method implemented"
    if hasattr(A, "matMvTranspmult"):
        A.matMvTranspmult(x, y)
    else:
        for i in range(x.nvec):
            A.multTranspose(x[i], y[i])


def MvDSmatMult(X: MultiVector, A: np.array, Y: MultiVector) -> None:
    assert (
        X.nvec == A.shape[0]
    ), "X Number of vecs incompatible with number of rows in A"
    assert (
        Y.nvec == A.shape[1]
    ), "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec):
        Y[j].scale(0.0)
        X.reduce(Y[j], A[:, j].flatten())
