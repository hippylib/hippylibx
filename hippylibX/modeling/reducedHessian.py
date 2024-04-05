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
from .variables import STATE, PARAMETER, ADJOINT
import dolfinx as dlx
import petsc4py


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


class ReducedHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:
    - :code:`model`:               the object which contains the description of the problem.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.

    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """

    def __init__(self, model, misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.gauss_newton_approx = self.model.gauss_newton_approx
        self.misfit_only = misfit_only
        self.ncalls = 0

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat = model.generate_vector(STATE)
        self.phat = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.model.prior.M.getSizes(), comm=self.model.prior.Vh.mesh.comm
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        self.petsc_wrapper.destroy()

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        return self.petsc_wrapper

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`. Return the result in :code:`y`.
        """

        x_dlx = self.model.generate_vector(PARAMETER)
        y_dlx = self.model.generate_vector(PARAMETER)

        x_tmp_dlx = dlx.la.create_petsc_vector_wrap(x_dlx)
        x_tmp_dlx.axpy(1.0, x)

        if self.gauss_newton_approx:
            self.GNHessian(x_dlx, y_dlx)
        else:
            self.TrueHessian(x_dlx, y_dlx)

        tmp = dlx.la.create_petsc_vector_wrap(y_dlx)

        y.axpby(1.0, 0.0, tmp)  # y = 1. tmp + 0.*y
        tmp.destroy()
        x_tmp_dlx.destroy()

        self.ncalls += 1

    def GNHessian(self, x: dlx.la.Vector, y: dlx.la.Vector) -> None:
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)

        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.array[:] += self.yhelp.array

    def TrueHessian(self, x: dlx.la.Vector, y: dlx.la.Vector) -> None:
        """
        Apply the the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)
        self.rhs_adj.array[:] = self.rhs_adj.array + (-1.0) * self.rhs_adj2.array
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.array[:] += self.yhelp.array
        self.model.applyWmu(self.uhat, self.yhelp)
        y.array[:] -= self.yhelp.array
        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.array[:] += self.yhelp.array
