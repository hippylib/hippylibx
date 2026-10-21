# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# Finite-displacement elasticity example with a St. Venant-Kirchhoff material.
# The forward problem is nonlinear: the displacement u solves
#     -div( F S ) = f in Omega, u = 0 on the boundary,
# with F = I + grad(u), E_GL = (F^T F - I)/2, S = 2 mu E_GL + lambda tr(E_GL) I.
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


class StVenantKirchhoffElasticity:
    def __init__(self, nu: float, f: dlx.fem.Constant):
        self.nu = nu
        self.f = f
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(
        self, u: dlx.fem.Function, m: dlx.fem.Function, p: dlx.fem.Function
    ) -> ufl.form.Form:
        E = ufl.exp(m)
        mu = E / (2.0 * (1.0 + self.nu))
        lmbda = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        Id = ufl.Identity(len(u))
        F = Id + ufl.grad(u)
        E_GL = 0.5 * (F.T * F - Id)
        S = 2.0 * mu * E_GL + lmbda * ufl.tr(E_GL) * Id

        return ufl.inner(F * S, ufl.grad(p)) * self.dx - ufl.inner(self.f, p) * self.dx


def run_inversion(
    nx: int, ny: int, n_obs_1d: int, noise_level: float, reg_param: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    sep = "\n" + "#" * 80 + "\n"
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    msh = dlx.mesh.create_unit_square(comm, nx, ny)
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

    # Homogeneous Dirichlet B.C. on the whole boundary
    fdim = msh.topology.dim - 1
    boundary_facets = dlx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True)
    )
    dirichlet_dofs = dlx.fem.locate_dofs_topological(Vh_u, fdim, boundary_facets)
    u0 = np.zeros(gdim, dtype=dlx.default_scalar_type)
    bc = dlx.fem.dirichletbc(u0, dirichlet_dofs, Vh_u)

    # FORWARD MODEL
    nu = 0.3
    f = dlx.fem.Constant(msh, dlx.default_scalar_type((0.0, -0.5)))
    pde_handler = StVenantKirchhoffElasticity(nu, f)
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [bc], [bc], is_fwd_linear=False)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)
    m_true.interpolate(
        lambda x: np.log(
            1.0 + 2.0 * (((x[0] - 0.5) ** 2 + (x[1] - 0.6) ** 2) ** 0.5 < 0.2)
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
    # n_obs_1d x n_obs_1d grid of points that does not touch the boundary
    xx = (np.arange(n_obs_1d) + 0.5) / n_obs_1d
    X, Y = np.meshgrid(xx, xx)
    obs_points = np.column_stack([X.ravel(), Y.ravel()])
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
    m0.interpolate(lambda x: 0.5 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
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
        "hyperelasticity_np{0:d}.bp".format(nproc),
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
    nx = 128
    ny = 128
    n_obs_1d = 64
    noise_level = 0.01
    reg_param = {"gamma": 50.0, "delta": 0.5}
    final_results = run_inversion(nx, ny, n_obs_1d, noise_level, reg_param)

    k, d = (
        final_results["eigen_decomposition_results"]["k"],
        final_results["eigen_decomposition_results"]["d"],
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        plt.savefig("hyperelasticity_result_FD_Gradient_Hessian_Check")
        plt.figure()
        plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
        plt.yscale("log")
        plt.xlabel("index")
        plt.ylabel("eigenvalue")
        plt.title("Generalized eigenvalues of the data misfit Hessian at the MAP")
        plt.savefig("hyperelasticity_Eigen_Decomposition_results.png")
