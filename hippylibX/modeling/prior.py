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

import dolfinx as dlx
import ufl  # type: ignore
import numpy as np
import petsc4py

# decorator for functions in classes that are not used -> may not be needed in the final
# version of X


def unused_function(func):
    return None


class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):  # type: ignore
        self.A = A  # should be petsc4py.PETSc.Mat
        self.Msolver = Msolver

        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

        self.petsc_wrapper = petsc4py.PETSc.Mat().createPython(  # type: ignore
            self.A.getSizes(), comm=self.A.getComm()
        )
        self.petsc_wrapper.setPythonContext(self)
        self.petsc_wrapper.setUp()

    def __del__(self):
        self.petsc_wrapper.destroy()

    @property
    def mat(self):
        return self.petsc_wrapper

    def mpi_comm(self):
        return self.A.comm

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:  # type: ignore
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


class _BilaplacianRsolver:
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):  # type: ignore
        self.Asolver = Asolver
        self.M = M
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    def solve(self, b: petsc4py.PETSc.Vec, x: petsc4py.PETSc.Vec) -> int:  # type: ignore
        self.Asolver.solve(b, self.help1)
        nit = self.Asolver.its
        self.M.mult(self.help1, self.help2)
        self.Asolver.solve(self.help2, x)
        nit += self.Asolver.its
        return nit


class SqrtPrecisionPDE_Prior:
    def __init__(
        self,
        Vh: dlx.fem.FunctionSpace,
        sqrt_precision_varf_handler,
        mean=None,  # type: ignore
    ):
        """
        Construct the prior model.
        Input:
        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """

        """
        This class implement a prior model with covariance matrix
        :math:`C = A^{-1} M A^-1`,
        where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
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

        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))  # type: ignore
        self.M.assemble()

        self.Msolver = self._createsolver(self.petsc_options_M)
        self.Msolver.setOperators(self.M)

        self.A = dlx.fem.petsc.assemble_matrix(  # type: ignore
            dlx.fem.form(sqrt_precision_varf_handler(trial, test))
        )
        self.A.assemble()
        self.Asolver = self._createsolver(self.petsc_options_A)

        if self.petsc_options_A["pc_type"] == "hypre":
            pc = self.Asolver.getPC()
            pc.setHYPREType("boomeramg")

        self.Asolver.setOperators(self.A)

        qdegree = 2 * Vh._ufl_element.degree()  # type: ignore
        metadata = {"quadrature_degree": qdegree}

        num_sub_spaces = Vh.num_sub_spaces  # type: ignore

        if num_sub_spaces <= 1:  # SCALAR PARAMETER
            element = ufl.FiniteElement(
                "Quadrature", Vh.mesh.ufl_cell(), qdegree, quad_scheme="default"
            )

        else:  # Vector FIELD PARAMETER
            element = ufl.VectorElement(
                "Quadrature",
                Vh.mesh.ufl_cell(),
                qdegree,
                dim=num_sub_spaces,
                quad_scheme="default",
            )

        self.Qh = dlx.fem.FunctionSpace(Vh.mesh, element)

        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)

        Mqh = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, qh) * ufl.dx(metadata=metadata))
        )
        Mqh.assemble()

        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones, dMqh)
        dMqh.setArray(ones.getArray() / np.sqrt(dMqh.getArray()))
        Mqh.setDiagonal(dMqh)

        MixedM = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, test) * ufl.dx(metadata=metadata))
        )
        MixedM.assemble()

        self.sqrtM = MixedM.matMult(Mqh)

        self._R = _BilaplacianR(self.A, self.Msolver)

        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        self.mean = mean

        if self.mean is None:
            self.mean = self.generate_parameter(0)

    @property
    def R(self):
        return self._R.mat

    def generate_parameter(self, dim: int) -> dlx.la.Vector:
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.
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
        temp_petsc_vec_noise = dlx.la.create_petsc_vector_wrap(noise)

        rhs = self.sqrtM.createVecLeft()
        self.sqrtM.mult(temp_petsc_vec_noise, rhs)
        temp_petsc_vec_noise.destroy()

        temp_petsc_vec_s = dlx.la.create_petsc_vector_wrap(s)
        self.Asolver.solve(rhs, temp_petsc_vec_s)

        if add_mean:
            temp_petsc_vec_mean = dlx.la.create_petsc_vector_wrap(self.mean)
            temp_petsc_vec_s.axpy(1.0, temp_petsc_vec_mean)
            temp_petsc_vec_mean.destroy()

        rhs.destroy()
        temp_petsc_vec_s.destroy()

    def _createsolver(self, petsc_options) -> petsc4py.PETSc.KSP:
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
        temp_petsc_vec_d = dlx.la.create_petsc_vector_wrap(self.mean).copy()
        temp_petsc_vec_m = dlx.la.create_petsc_vector_wrap(m)
        temp_petsc_vec_d.axpy(-1.0, temp_petsc_vec_m)
        temp_petsc_vec_Rd = dlx.la.create_petsc_vector_wrap(self.generate_parameter(0))

        # mult used, so need to have petsc4py Vec objects
        self.R.mult(temp_petsc_vec_d, temp_petsc_vec_Rd)

        return_value = 0.5 * temp_petsc_vec_Rd.dot(temp_petsc_vec_d)
        temp_petsc_vec_d.destroy()
        temp_petsc_vec_m.destroy()
        temp_petsc_vec_Rd.destroy()

        return return_value

    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> None:
        temp_petsc_vec_d = dlx.la.create_petsc_vector_wrap(m).copy()
        temp_petsc_vec_self_mean = dlx.la.create_petsc_vector_wrap(self.mean)
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)
        temp_petsc_vec_d.axpy(-1.0, temp_petsc_vec_self_mean)
        self.R.mult(temp_petsc_vec_d, temp_petsc_vec_out)
        temp_petsc_vec_d.destroy()
        temp_petsc_vec_self_mean.destroy()
        temp_petsc_vec_out.destroy()

    def setLinearizationPoint(
        self, m: dlx.la.Vector, gauss_newton_approx=False
    ) -> None:
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
        trial: ufl.TrialFunction, test: ufl.TestFunction
    ) -> ufl.form.Form:
        if Theta is None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx(
                metadata={"quadrature_degree": 4}
            )
        else:
            varfL = ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * ufl.dx(
                metadata={"quadrature_degree": 4}
            )

        varfM = ufl.inner(trial, test) * ufl.dx

        varf_robin = ufl.inner(trial, test) * ufl.ds

        if robin_bc:
            robin_coeff = gamma * ufl.sqrt(delta / gamma) / 1.42

        else:
            robin_coeff = 0.0

        return gamma * varfL + delta * varfM + robin_coeff * varf_robin

    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean)
