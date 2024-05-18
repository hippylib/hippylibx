# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import dolfinx as dlx
import math
from .variables import STATE, PARAMETER, ADJOINT
from typing import Any


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


class Model:
    """
    This class contains the full description of the inverse problem.
    As inputs it takes a :code:``PDEProblem object`, a :code:`Prior` object, and a :code:`Misfit` object.

    In the following we will denote with

        - :code:`u` the state variable
        - :code:`m` the (model) parameter variable
        - :code:`p` the adjoint variable
    """

    def __init__(self, problem, prior, misfit):
        """
        Create a model given:
            - problem: the description of the forward/adjoint problem and all the sensitivities
            - prior: the prior component of the cost functional
            - misfit: the misfit componenent of the cost functional
        """

        self.problem = problem
        self.prior = prior
        self.misfit = misfit
        self.gauss_newton_approx = False

        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

    def generate_vector(self, component="ALL"):
        """
        By default, return the list :code:`[u,m,p]` where:
        
            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
        
        If :code:`component = STATE` return only :code:`u`
            
        If :code:`component = PARAMETER` return only :code:`m`
            
        If :code:`component = ADJOINT` return only :code:`p`
        """
        if component == "ALL":
            x = [
                self.problem.generate_state(),
                self.problem.generate_parameter(),
                self.problem.generate_state(),
            ]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = self.problem.generate_parameter()
        elif component == ADJOINT:
            x = self.problem.generate_state()

        return x

    def init_parameter(self, m) -> dlx.la.Vector:
        """
        Reshape :code:`m` so that it is compatible with the parameter variable
        """
        return self.problem.generate_parameter()

    def cost(self, x: list) -> list:
        """
        Given the list :code:`x = [u,m,p]` which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of
        the misfit functional and the regularization functional.

        Return the list [cost functional, regularization functional, misfit functional]

        .. note:: :code:`p` is not needed to compute the cost functional
        """
        misfit_cost = self.misfit.cost(x)
        reg_cost = self.prior.cost(x[PARAMETER])

        return [misfit_cost + reg_cost, reg_cost, misfit_cost]

    def solveFwd(self, out: dlx.la.Vector, x: list) -> None:
        """
        Solve the (possibly non-linear) forward problem.

        Parameters:

            - :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
            - :code:`x = [u,m,p]` provides

                1) the parameter variable :code:`m` for the solution of the forward problem
                2) the initial guess :code:`u` if the forward problem is non-linear
                .. note:: :code:`p` is not accessed.
        """
        self.n_fwd_solve = self.n_fwd_solve + 1
        self.problem.solveFwd(out, x)

    def solveAdj(self, out: dlx.la.Vector, x: list) -> None:
        """
        Solve the linear adjoint problem.

        Parameters:

            - :code:`out`: is the solution of the adjoint problem (i.e. the adjoint :code:`p`) (Output parameter)
            - :code:`x = [u, m, p]` provides

                1) the parameter variable :code:`m` for assembling the adjoint operator
                2) the state variable :code:`u` for assembling the adjoint right hand side
                .. note:: :code:`p` is not accessed
        """

        self.n_adj_solve = self.n_adj_solve + 1
        rhs = self.problem.generate_state()
        self.misfit.grad(STATE, x, rhs)

        rhs.array[:] *= -1.0
        self.problem.solveAdj(out, x, rhs)

    def evalGradientParameter(
        self, x: list, mg: dlx.la.Vector, misfit_only=False
    ) -> float:
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u,m,p]`.

        Parameters:
            - :code:`x = [u,m,p]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)`, mtest being a test function in the parameter space \
            (Output parameter)
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """

        tmp = self.generate_vector(PARAMETER)

        self.problem.evalGradientParameter(x, mg)

        self.misfit.grad(PARAMETER, x, tmp)

        mg.petsc_vec.axpy(1.0, tmp.petsc_vec)

        if not misfit_only:
            self.prior.grad(x[PARAMETER], tmp)

            mg.petsc_vec.axpy(1.0, tmp.petsc_vec)

        self.prior.Msolver.solve(mg.petsc_vec, tmp.petsc_vec)

        return math.sqrt(mg.petsc_vec.dot(tmp.petsc_vec))

    def setPointForHessianEvaluations(self, x: list, gauss_newton_approx=False) -> None:
        """
        Specify the point :code:`x = [u,m,p]` at which the Hessian operator (or the Gauss-Newton approximation)
        needs to be evaluated.
        Parameters:
            - :code:`x = [u,m,p]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.
            - :code:`gauss_newton_approx (bool)`: whether to use Gauss-Newton approximation (default: use Newton)
        .. note:: This routine should either:
            - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
            - or partially precompute the block of the hessian (if feasible)
        """

        self.gauss_newton_approx = gauss_newton_approx
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)
        self.prior.setLinearizationPoint(x[PARAMETER], self.gauss_newton_approx)

    def solveFwdIncremental(self, sol: dlx.la.Vector, rhs: dlx.la.Vector) -> None:
        """
        Solve the linearized (incremental) forward problem for a given right-hand side
        Parameters:
            - :code:`sol` the solution of the linearized forward problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol, rhs, False)

    def solveAdjIncremental(self, sol: dlx.la.Vector, rhs: dlx.la.Vector) -> None:
        """
        Solve the incremental adjoint problem for a given right-hand side

        Parameters:

            - :code:`sol` the solution of the incremental adjoint problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol, rhs, True)

    def applyC(self, dm: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
        :code:`out` = :math:`C dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`C` block on :code:`dm`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """

        self.problem.apply_ij(ADJOINT, PARAMETER, dm, out)

    def applyCt(self, dp: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C^t dp`

        Parameters:

            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C^T` block on :code:`dp`

        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER, ADJOINT, dp, out)

    def applyWuu(self, du: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the :math:`W_{uu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{uu} du`

        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{uu}` block on :code:`du`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.misfit.apply_ij(STATE, STATE, du, out)
        if not self.gauss_newton_approx:
            tmp = self.generate_vector(STATE)
            self.problem.apply_ij(STATE, STATE, du, tmp)

            out.array[:] += tmp.array

    def applyWum(self, dm: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the :math:`W_{um}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{um} dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{um}` block on :code:`du`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """

        if self.gauss_newton_approx:
            out.array[:] = 0.0
        else:
            self.problem.apply_ij(STATE, PARAMETER, dm, out)
            tmp = self.generate_vector(STATE)
            self.misfit.apply_ij(STATE, PARAMETER, dm, tmp)
            out.array[:] += tmp.array

    def applyWmu(self, du: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the :math:`W_{mu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{mu} du`

        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{mu}` block on :code:`du`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.array[:] = 0.0
        else:
            self.problem.apply_ij(PARAMETER, STATE, du, out)
            tmp = self.generate_vector(PARAMETER)

            self.misfit.apply_ij(PARAMETER, STATE, du, tmp)
            out.array[:] += tmp.array

    def applyR(self, dm: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the regularization :math:`R` to a (incremental) parameter variable.
        :code:`out` = :math:`R dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of :math:`R` on :code:`dm`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.prior.R.mult(dm.petsc_vec, out.petsc_vec)

    def Rsolver(self) -> Any:
        """
        Return an object :code:`Rsovler` that is a suitable solver for the regularization
        operator :math:`R`.

        The solver object should implement the method :code:`Rsolver.solve(z,r)` such that
        :math: `Rz approx r`
        """
        return self.prior.Rsolver

    def applyWmm(self, dm: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{mm}` on block :code:`dm`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.array[:] = 0.0
        else:
            self.problem.apply_ij(PARAMETER, PARAMETER, dm, out)
            tmp = self.generate_vector(PARAMETER)
            self.misfit.apply_ij(PARAMETER, PARAMETER, dm, tmp)

            out.array[:] += tmp.array
