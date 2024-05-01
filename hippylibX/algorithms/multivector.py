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
    def __init__(self, example_vec: petsc4py.PETSc.Vec, nvec: int):
        self.nvec = nvec
        self.data = []
        for i in range(self.nvec):
            self.data.append(example_vec.duplicate())

    @classmethod
    def createFromVec(
        cls, example_vec: petsc4py.PETSc.Vec, nvec: int
    ) -> Type["MultiVector"]:
        return cls(example_vec, nvec)

    @classmethod
    def createFromMultiVec(cls, mv: Type["MultiVector"]) -> Type["MultiVector"]:
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
        if isinstance(alpha, float):
            for d in self.data:
                d.scale(alpha)
        else:
            for i, d in enumerate(self.data):
                d.scale(alpha[i])

    def dot(self, v: Union[petsc4py.PETSc.Vec, Type["MultiVector"]]) -> np.array:
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
        for i in range(self.nvec):
            y.axpy(alpha[i], self[i])

    def axpy(self, alpha: Union[float, np.array], Y: Type["MultiVector"]) -> None:
        if isinstance(alpha, float):
            for i in range(self.nvec):
                self[i].axpy(alpha, Y[i])
        else:
            for i in range(self.nvec):
                self[i].axpy(alpha[i], Y[i])

    def norm(self, norm_type: petsc4py.PETSc.NormType) -> np.array:
        norm_vals = np.zeros(self.nvec)
        for i in range(self.nvec):
            norm_vals[i] = self[i].norm(norm_type)
        return norm_vals

    def Borthogonalize(self, B: petsc4py.PETSc.Mat):
        return self._mgs_stable(B)

    def _mgs_stable(
        self, B: petsc4py.PETSc.Mat
    ) -> tuple[Type["MultiVector"], np.array]:
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
