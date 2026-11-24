# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# Finite-displacement elasticity example with a St. Venant-Kirchhoff material
# and sliding (normal-only) boundary conditions weakly enforced with Nitsche's
# method. The forward problem is nonlinear: the displacement u solves
#     -div( P ) = f       in Omega,
#     u.n = g             on the boundary,
#     (P n) - (n.P n) n = 0  on the boundary (zero tangential traction),
# with F = I + grad(u), E_GL = (F^T F - I)/2, S = 2 mu E_GL + lambda tr(E_GL) I,
# P = F S, and n the outward unit normal (in the reference configuration).
# The normal displacement g = u_D.n is obtained from a prescribed displacement
# field u_D. The boundary condition is imposed through facet integrals using
# ufl.FacetNormal, so the code works in 2D and 3D and for boundaries that are
# not aligned with the coordinate axes.
# The inversion parameter is m = log(E), where E is the Young modulus.
# The Poisson ratio is known. Displacement data are collected on a uniform
# grid of points in the interior of the domain. H1 Tikhonov regularization.
import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import sys
import os
import dolfinx.fem.petsc
from matplotlib import pyplot as plt
from typing import Dict

sys.path.append(os.environ.get("HIPPYLIBX_BASE_DIR", "../"))
import hippylibX as hpx


class StVenantKirchhoffElasticityNitsche:
    def __init__(
        self,
        msh: dlx.mesh.Mesh,
        nu: float,
        f: dlx.fem.Constant,
        u_D: ufl.core.expr.Expr,
        beta: float,
    ):
        self.nu = nu
        self.f = f
        self.beta = beta
        self.n = ufl.FacetNormal(msh)
        self.h = ufl.CellDiameter(msh)
        # Prescribed normal displacement
        self.g = ufl.dot(u_D, self.n)
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

    def __call__(
        self, u: dlx.fem.Function, m: dlx.fem.Function, p: dlx.fem.Function
    ) -> ufl.form.Form:
        E = ufl.exp(m)
        mu = E / (2.0 * (1.0 + self.nu))
        lmbda = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        Id = ufl.Identity(len(u))
        n = self.n

        # First Piola-Kirchhoff stress
        F = Id + ufl.grad(u)
        E_GL = 0.5 * (F.T * F - Id)
        S = 2.0 * mu * E_GL + lmbda * ufl.tr(E_GL) * Id
        P = F * S

        # Directional derivative of P with respect to u in the direction p
        # (used for the symmetric Nitsche term)
        dF = ufl.grad(p)
        dE_GL = 0.5 * (dF.T * F + F.T * dF)
        dS = 2.0 * mu * dE_GL + lmbda * ufl.tr(dE_GL) * Id
        dP = dF * S + F * dS

        gap = ufl.dot(u, n) - self.g
        penalty = self.beta * (2.0 * mu + lmbda) / self.h

        return (
            ufl.inner(P, ufl.grad(p)) * self.dx
            - ufl.inner(self.f, p) * self.dx
            - ufl.dot(n, P * n) * ufl.dot(p, n) * self.ds
            - ufl.dot(n, dP * n) * gap * self.ds
            + penalty * gap * ufl.dot(p, n) * self.ds
        )


def run_inversion(
    msh: dlx.mesh.Mesh,
    n_obs_1d: int,
    noise_level: float,
    reg_param: Dict[str, float],
    compute_eigendecomposition: bool = True,
) -> Dict[str, Dict[str, float]]:
    sep = "\n" + "#" * 80 + "\n"
    comm = msh.comm
    rank = comm.rank
    nproc = comm.size

    gdim = msh.geometry.dim
    Vh_u = dlx.fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vh = [Vh_u, Vh_m, Vh_u]

    ndofs = [
        Vh_u.dofmap.index_map.size_global * Vh_u.dofmap.index_map_bs,
        Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
    ]
    hpx.master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
    hpx.master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

    # FORWARD MODEL
    nu = 0.3
    # Body force pointing in the negative direction of the last coordinate axis
    f_val = np.zeros(gdim, dtype=dlx.default_scalar_type)
    # f_val[-1] = -0.5
    f = dlx.fem.Constant(msh, f_val)

    # Prescribed displacement on the boundary: uniform 2% expansion about the
    # center of the domain. Only its normal component u_D.n is enforced.
    x_coord = ufl.SpatialCoordinate(msh)
    x_center = dlx.fem.Constant(msh, np.full(gdim, 0.5, dtype=dlx.default_scalar_type))
    u_D = 0.02 * (x_coord - x_center)

    beta = 100.0  # Nitsche penalty parameter
    pde_handler = StVenantKirchhoffElasticityNitsche(msh, nu, f, u_D, beta)
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [], is_fwd_linear=False)

    # GROUND TRUTH
    # Stiff spherical (circular in 2D) inclusion
    c_incl = np.array([0.5, 0.6, 0.5])[:gdim]
    m_true = dlx.fem.Function(Vh_m)
    m_true.interpolate(
        lambda x: np.log(
            1.0 + 2.0 * (np.linalg.norm(x[:gdim] - c_incl.reshape(-1, 1), axis=0) < 0.2)
        )
    )
    m_true.x.scatter_forward()

    m_true = m_true.x
    u_true = pde.generate_state()

    x_true = [u_true, m_true, None]
    pde.solveFwd(u_true, x_true)

    nloc = Vh_u.dofmap.index_map.size_local
    u_mag = np.linalg.norm(u_true.array[: nloc * gdim].reshape(-1, gdim), axis=1)
    u_max = comm.allreduce(np.max(u_mag, initial=0.0), op=MPI.MAX)
    hpx.master_print(comm, "Maximum displacement: {0:.4e}".format(u_max))

    # LIKELIHOOD
    # Uniform grid of n_obs_1d points per direction that does not touch the boundary
    xx = (np.arange(n_obs_1d) + 0.5) / n_obs_1d
    grids = np.meshgrid(*([xx] * gdim), indexing="ij")
    obs_points = np.column_stack([g.ravel() for g in grids])
    B = hpx.pointwiseInterpolationMatrix(Vh_u, obs_points)
    data = B.createVecLeft()
    B.mult(u_true.petsc_vec, data)

    noise_std_dev = noise_level * u_max
    noise_variance = noise_std_dev**2
    hpx.master_print(comm, "Noise standard deviation: {0:.4e}".format(noise_std_dev))
    hpx.parRandom.normal_perturb(noise_std_dev, data)
    misfit = hpx.DiscreteStateObservationMisfit(B, data, noise_variance)

    # REGULARIZATION
    m_ref = dlx.fem.Function(Vh_m)
    m_ref.x.array[:] = 0.0

    reg_handler = hpx.H1TikhonvFunctional(reg_param["gamma"], reg_param["delta"], m_ref)
    reg = hpx.VariationalRegularization(Vh_m, reg_handler)

    model = hpx.Model(pde, reg, misfit)

    m0 = dlx.fem.Function(Vh_m)
    m0.interpolate(lambda x: 0.5 * np.prod(np.sin(np.pi * x[:gdim]), axis=0))
    m0.x.scatter_forward()
    m0 = m0.x

    # Keep the finite difference perturbations small, so that the perturbed
    # Young modulus remains in the range where the nonlinear solver converges
    eps = 1e-2 * np.power(0.5, np.arange(16))[::-1]

    data_misfit_True = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=True, verbose=(rank == 0), eps=eps
    )

    data_misfit_False = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=False, verbose=(rank == 0), eps=eps
    )

    # # #######################################

    initial_guess_m = pde.generate_parameter()
    initial_guess_m.array[:] = m_ref.x.array[:]
    x = [
        model.generate_vector(hpx.STATE),
        initial_guess_m,
        model.generate_vector(hpx.ADJOINT),
    ]
    if rank == 0:
        print(sep, "Find the MAP point", sep)

    parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-6
    parameters["abs_tolerance"] = 1e-9
    parameters["max_iter"] = 500
    parameters["cg_coarse_tolerance"] = 5e-1
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1

    solver = hpx.ReducedSpaceNewtonCG(model, parameters)

    x = solver.solve(x)

    if solver.converged:
        hpx.master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        hpx.master_print(comm, "\nNot Converged")

    hpx.master_print(
        comm, "Termination reason: ", solver.termination_reasons[solver.reason]
    )
    hpx.master_print(comm, "Final gradient norm: ", solver.final_grad_norm)
    hpx.master_print(comm, "Final cost: ", solver.final_cost)

    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER], name="m_map")
    m_true_fun = hpx.vector2Function(m_true, Vh[hpx.PARAMETER], name="m_true")

    V_P1 = dlx.fem.functionspace(msh, ("Lagrange", 1, (gdim,)))

    u_true_fun = dlx.fem.Function(V_P1, name="u_true")
    u_true_fun.interpolate(hpx.vector2Function(u_true, Vh[hpx.STATE]))
    u_true_fun.x.scatter_forward()

    u_map_fun = dlx.fem.Function(V_P1, name="u_map")
    u_map_fun.interpolate(hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE]))
    u_map_fun.x.scatter_forward()

    with dlx.io.VTXWriter(
        msh.comm,
        "hyperelasticity_nitsche_{0:d}d_np{1:d}.bp".format(gdim, nproc),
        [m_fun, m_true_fun, u_map_fun, u_true_fun],
    ) as vtx:
        vtx.write(0.0)

    optimizer_results = {}
    if (
        solver.termination_reasons[solver.reason]
        == "Norm of the gradient less than tolerance"
    ):
        optimizer_results["optimizer"] = True
    else:
        optimizer_results["optimizer"] = False

    # GENERALIZED EIGENDECOMPOSITION OF THE DATA MISFIT HESSIAN AT THE MAP POINT
    eigen_decomposition_results = None
    if compute_eigendecomposition:
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        Hmisfit = hpx.ReducedHessian(model, misfit_only=True)

        k = 80
        p = 20
        hpx.master_print(
            comm,
            "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(
                k, p
            ),
        )

        Omega = hpx.MultiVector(x[hpx.PARAMETER].petsc_vec, k + p)
        hpx.parRandom.normal(1.0, Omega)

        d, U = hpx.doublePassG(Hmisfit.mat, reg.R, reg.Rsolver, Omega, k, s=1)

        eigen_decomposition_results = {"A": Hmisfit, "B": reg, "k": k, "d": d, "U": U}

    final_results = {
        "data_misfit_True": data_misfit_True,
        "data_misfit_False": data_misfit_False,
        "optimizer_results": optimizer_results,
        "eigen_decomposition_results": eigen_decomposition_results,
    }

    B.destroy()
    data.destroy()

    return final_results

    #######################################


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    dim = 2
    noise_level = 0.01
    reg_param = {"gamma": 50.0, "delta": 0.5}
    if dim == 2:
        msh = dlx.mesh.create_unit_square(comm, 128, 128)
        n_obs_1d = 64
    else:
        msh = dlx.mesh.create_unit_cube(comm, 16, 16, 16)
        n_obs_1d = 16
    final_results = run_inversion(msh, n_obs_1d, noise_level, reg_param)

    k, d = (
        final_results["eigen_decomposition_results"]["k"],
        final_results["eigen_decomposition_results"]["d"],
    )
    if comm.rank == 0:
        plt.savefig("hyperelasticity_nitsche_result_FD_Gradient_Hessian_Check")
        plt.figure()
        plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigenvalue")
        plt.title("Generalized eigenvalues of the data misfit Hessian at the MAP")
        plt.savefig("hyperelasticity_nitsche_Eigen_Decomposition_results.png")
