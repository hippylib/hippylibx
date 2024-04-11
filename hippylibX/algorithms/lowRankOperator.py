# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import petsc4py.PETSc
from .multivector import MultiVector, MatMvMult
import numpy as np
import petsc4py


class LowRankOperator:
    """
    This class model the action of a low rank operator :math:`A = U D U^T`.
    Here :math:`D` is a diagonal matrix, and the columns of are orthonormal
    in some weighted inner-product.
    """

    def __init__(
        self, d: np.array, U: MultiVector, createVecLeft=None, createVecRight=None
    ):
        """
        Construct the low rank operator given :code:`d` and :code:`U`.
        """
        self.d = d
        self.U = U
        self.createVecLeft = createVecLeft
        self.createVecRight = createVecRight

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        """
        Compute :math:`y = Ax = U D U^T x`
        """
        Utx = self.U.dot(x)
        dUtx = self.d * Utx  # elementwise mult
        self.U.reduce(y, dUtx)

    def solve(self, rhs: petsc4py.PETSc.Vec, sol: petsc4py.PETSc.Vec) -> None:
        """
        Compute :math:`\mbox{sol} = U D^-1 U^T x`
        """
        Utr = self.U.dot(rhs)
        dinvUtr = Utr / self.d
        self.U.reduce(sol, dinvUtr)

    def get_diagonal(self, diag: petsc4py.PETSc.Vec) -> None:
        """
        Compute the diagonal of :code:`A`.
        """
        diag.scale(0.0)
        tmp = self.U[0].duplicate()
        for i in range(self.U.nvec):           
            tmp.pointwiseMult(self.U[i], self.U[i])
            diag.axpy(self.d[i], tmp)

    def trace(self, W=None) -> float:
        """
        Compute the trace of :code:`A`.
        If the weight :code:`W` is given, compute the trace of :math:`W^{1/2} A W^{1/2}`.
        This is equivalent to :math:`\mbox{tr}_W(A) = \sum_i \lambda_i`,
        where :math:`\lambda_i` are the generalized eigenvalues of
        :math:`A x = \lambda W^{-1} x`.

        .. note:: If :math:`U` is a :math:`W`-orthogonal matrix then :math:`\mbox{tr}_W(A) = \sum_i D(i,i)`.
        """
        if W is None:
            tmp = self.U[0].duplicate()
            tmp.scale(0.0)
            self.U.reduce(tmp, np.sqrt(self.d))
            tr = tmp.dot(tmp)
        else:
            WU = MultiVector.createFromVec(self.U[0], self.U.nvec)
            MatMvMult(W, self.U, WU)
            diagWUtU = np.zeros_like(self.d)
            for i in range(self.d.shape[0]):
                diagWUtU[i] = WU[i].dot(self.U[i])
            tr = np.sum(self.d * diagWUtU)

        return tr
