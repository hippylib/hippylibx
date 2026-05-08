# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import hippylibX as hpx
import dolfinx as dlx
import ufl
from mpi4py import MPI
import petsc4py


class DiscreteStateObservationMisfit(object):
    r"""
    This class define a misfit function for a discrete linear observation operator B
    math: 1/(2\sigma^2) || B u - d ||^2
    """

    def __init__(self, B: petsc4py.PETSc.Mat, d: petsc4py.PETSc.Mat, sigma2: float):
        r"""
        Constructor:
        :code:`B` is the observation operator
        :code:`d` is the data
        :code:`sigma2` is the variance of the noise
        """
        self.B = B
        self.d = d
        self.sigma2 = sigma2
        self.y = self.B.createVecLeft()
        self.s = self.B.createVecRight()

    def cost(self, x: list) -> float:
        self.B.mult(x[hpx.STATE].petsc_vec, self.y)
        self.y.axpy(-1.0, self.d)
        return (0.5 / self.sigma2) * self.y.dot(self.y)

    def grad(self, i: int, x: list, out: dlx.la.Vector) -> None:
        if i == hpx.STATE:
            self.B.mult(x[hpx.STATE].petsc_vec, self.y)
            self.y.axpy(-1.0, self.d)
            self.y.scale(1.0 / self.sigma2)
            self.B.multTranspose(self.y, out.petsc_vec)
        elif i == hpx.PARAMETER:
            out.petsc_vec.zeroEntries()
        else:
            raise (i)

    def setLinearizationPoint(self, x: list, gauss_newton_approx=False) -> None:
        pass

    def apply_ij(self, i: int, j: int, dir: dlx.la.Vector, out: dlx.la.Vector) -> None:
        r"""
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        if i == hpx.STATE and j == hpx.STATE:
            self.B.mult(dir.petsc_vec, self.y)
            self.y.scale(1.0 / self.sigma2)
            self.B.multTranspose(self.y, out.petsc_vec)
        else:
            out.petsc_vec.zeroEntries()


class NonGaussianContinuousMisfit(object):
    """
    Abstract class to model the misfit component of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.

    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed.
    """

    def __init__(self, Vh: list, form, bc0=[]):
        self.Vh = Vh
        self.form = form
        self.bc0 = bc0

        self.x_lin_fun = None
        self.x_test = [
            ufl.TestFunction(Vh[hpx.STATE]),
            ufl.TestFunction(Vh[hpx.PARAMETER]),
        ]
        self.gauss_newton_approx = False

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    def cost(self, x: list) -> float:
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed.
        """
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        loc_cost = self.form(u_fun, m_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.Vh[hpx.STATE].mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM)

    def grad(self, i: int, x: list, out: dlx.la.Vector) -> None:
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        x_fun = [u_fun, m_fun]

        out.array[:] = 0.0

        dlx.fem.petsc.assemble_vector(
            out.petsc_vec,
            dlx.fem.form(ufl.derivative(self.form(*x_fun), x_fun[i], self.x_test[i])),
        )
        out.petsc_vec.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES, petsc4py.PETSc.ScatterMode.REVERSE
        )
        dlx.fem.petsc.set_bc(out.petsc_vec, self.bc0)

    def setLinearizationPoint(self, x: list, gauss_newton_approx=False) -> None:
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]
        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx

    def apply_ij(self, i: int, j: int, dir: dlx.la.Vector, out: dlx.la.Vector) -> None:
        r"""
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        form = self.form(*self.x_lin_fun)
        if j == hpx.STATE:
            dlx.fem.petsc.set_bc(dir.petsc_vec, self.bc0)

        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = dlx.fem.form(
            ufl.derivative(
                ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]),
                self.x_lin_fun[j],
                dir_fun,
            )
        )
        out.array[:] = 0.0
        dlx.fem.petsc.assemble_vector(out.petsc_vec, action)
        out.petsc_vec.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES, petsc4py.PETSc.ScatterMode.REVERSE
        )
        if i == hpx.STATE:
            dlx.fem.petsc.set_bc(out.petsc_vec, self.bc0)
