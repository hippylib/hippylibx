# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import petsc4py
import petsc4py.PETSc
from mpi4py import MPI

import basix.ufl
import dolfinx as dlx
import numpy as np
import ufl


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):
        self.A = A  # should be petsc4py.PETSc.Mat
        self.Msolver = Msolver

        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(
            self.A.getSizes(),
            comm=self.A.getComm(),
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        self.petsc_wrapper.destroy()

    @property
    def mat(self) -> petsc4py.PETSc.Mat:
        return self.petsc_wrapper

    def mpi_comm(self) -> MPI.Intracomm:
        return self.A.comm

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


class _BilaplacianRsolver:
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):
        self.Asolver = Asolver
        self.M = M
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    def solve(self, b: petsc4py.PETSc.Vec, x: petsc4py.PETSc.Vec) -> int:
        self.Asolver.solve(b, self.help1)
        nit = self.Asolver.its
        self.M.mult(self.help1, self.help2)
        self.Asolver.solve(self.help2, x)
        nit += self.Asolver.its
        return nit

    def generate_vector(self) -> petsc4py.PETSc.Vec:
        return self.M.createVecLeft()


class SqrtPrecisionPDE_Prior:
    """
    This class implement a prior model with covariance matrix
    :math:`C = A^{-1} M A^-1`,
    where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
    """

    def __init__(
        self,
        Vh: dlx.fem.FunctionSpace,
        sqrt_precision_varf_handler,
        mean=None,
    ):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """

        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

        self.Vh = Vh
        self.sqrt_precision_varf_handler = sqrt_precision_varf_handler

        self.petsc_options_M = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        self.petsc_options_A = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }

        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)

        varfM = ufl.inner(trial, test) * self.dx

        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        self.Msolver = self._createsolver(self.petsc_options_M)
        self.Msolver.setOperators(self.M)

        self.A = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(sqrt_precision_varf_handler(trial, test)),
        )
        self.A.assemble()
        self.Asolver = self._createsolver(self.petsc_options_A)

        if self.petsc_options_A["pc_type"] == "hypre":
            pc = self.Asolver.getPC()
            pc.setHYPREType("boomeramg")

        self.Asolver.setOperators(self.A)

        qdegree = 2 * Vh._ufl_element.degree
        metadata = {"quadrature_degree": qdegree}

        num_sub_spaces = Vh.num_sub_spaces

        if num_sub_spaces <= 1:  # SCALAR PARAMETER
            element = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), degree=qdegree)

        else:  # Vector FIELD PARAMETER
            element = basix.ufl.element(
                "Lagrange",
                Vh.mesh.topology.cell_name(),
                degree=qdegree,
                shape=(num_sub_spaces,),
            )

        self.Qh = dlx.fem.functionspace(Vh.mesh, element)

        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)

        Mqh = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, qh) * ufl.dx(metadata=metadata)),
        )
        Mqh.assemble()

        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones, dMqh)
        dMqh.setArray(ones.getArray() / np.sqrt(dMqh.getArray()))
        Mqh.setDiagonal(dMqh)

        MixedM = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, test) * ufl.dx(metadata=metadata)),
        )
        MixedM.assemble()

        self.sqrtM = MixedM.matMult(Mqh)

        self._R = _BilaplacianR(self.A, self.Msolver)

        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        self.mean = mean

        if self.mean is None:
            self.mean = self.generate_parameter(0)

    @property
    def R(self) -> petsc4py.PETSc.Mat:
        return self._R.mat

    def generate_parameter(self, dim: int) -> dlx.la.Vector:
        """
        Initialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            return dlx.la.vector(self.Qh.dofmap.index_map)
        else:
            return dlx.la.vector(self.Vh.dofmap.index_map)

    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector, add_mean=True) -> None:
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """

        rhs = self.sqrtM.createVecLeft()

        self.sqrtM.mult(noise.petsc_vec, rhs)

        self.Asolver.solve(rhs, s.petsc_vec)

        if add_mean:
            s.petsc_vec.axpy(1.0, self.mean.petsc_vec)

        rhs.destroy()

    def _createsolver(self, petsc_options: dict) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create(self.Vh.mesh.comm)
        problem_prefix = f"dolfinx_solve_{id(self)}"
        ksp.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(problem_prefix)
        # petsc options for solver

        # Example:
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()

        return ksp

    def cost(self, m: dlx.la.Vector) -> float:
        d = self.mean.petsc_vec.copy()
        d.axpy(-1.0, m.petsc_vec)
        Rd = self.generate_parameter(0)
        self.R.mult(d, Rd.petsc_vec)
        return_value = 0.5 * Rd.petsc_vec.dot(d)
        d.destroy()
        return return_value

    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> None:
        d = m.petsc_vec.copy()
        d.axpy(-1.0, self.mean.petsc_vec)
        self.R.mult(d, out.petsc_vec)
        d.destroy()

    def setLinearizationPoint(self, m: dlx.la.Vector, gauss_newton_approx=False) -> None:
        return

    def __del__(self):
        self.Msolver.destroy()
        self.Asolver.destroy()
        self.M.destroy()
        self.A.destroy()
        self.sqrtM.destroy()


def BiLaplacianPrior(
    Vh: dlx.fem.FunctionSpace,
    gamma: float,
    delta: float,
    Theta=None,
    mean=None,
    robin_bc=False,
) -> SqrtPrecisionPDE_Prior:
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior`  with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.

    The magnitude of :math:`\\delta\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation lenght.

    Here :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.

    Input:

    - :code:`Vh`:              the finite element space for the parameter
    - :code:`gamma` and :code:`delta`: the coefficient in the PDE (floats, dl.Constant, dl.Expression, or dl.Function)
    - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
    - :code:`mean`:            the prior mean
    - :code:`rel_tol`:         relative tolerance for solving linear systems involving covariance matrix
    - :code:`max_iter`:        maximum number of iterations for solving linear systems involving covariance matrix
    - :code:`robin_bc`:        whether to use Robin boundary condition to remove boundary artifacts
    """

    def sqrt_precision_varf_handler(
        trial: ufl.TrialFunction,
        test: ufl.TestFunction,
    ) -> ufl.form.Form:
        if Theta is None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx(
                metadata={"quadrature_degree": 4},
            )
        else:
            varfL = ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * ufl.dx(
                metadata={"quadrature_degree": 4},
            )

        varfM = ufl.inner(trial, test) * ufl.dx

        varf_robin = ufl.inner(trial, test) * ufl.ds

        if robin_bc:
            robin_coeff = gamma * ufl.sqrt(delta / gamma) / 1.42

        else:
            robin_coeff = 0.0

        return gamma * varfL + delta * varfM + robin_coeff * varf_robin

    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean)
