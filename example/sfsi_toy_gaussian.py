# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# qpact problem with BiLaplacian Prior.
import os
import sys
from typing import Dict

from mpi4py import MPI

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np
import ufl
from matplotlib import pyplot as plt

sys.path.append(os.environ.get("HIPPYLIBX_BASE_DIR", "../"))
import hippylibX as hpx


class DiffusionApproximation:
    def __init__(self, D: float, u0: float):
        """
        Define the forward model for the diffusion approximation to radiative transfer equations

        D: diffusion coefficient 1/mu_eff with mu_eff = sqrt(3 mu_a (mu_a + mu_ps) ), where mu_a
           is the unknown absorption coefficient, and mu_ps is the reduced scattering coefficient

        u0: Incident fluence (Robin condition)

        ds: boundary integrator for Robin condition
        """
        self.D = D
        self.u0 = u0
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

    def __call__(
        self, u: dlx.fem.Function, m: dlx.fem.Function, p: dlx.fem.Function,
    ) -> ufl.form.Form:
        return (
            ufl.inner(self.D * ufl.grad(u), ufl.grad(p)) * ufl.dx(metadata={"quadrature_degree": 4})
            + ufl.exp(m) * ufl.inner(u, p) * self.dx
            + 0.5 * ufl.inner(u - self.u0, p) * self.ds
        )


class PACTMisfitForm:
    def __init__(self, d: float, sigma2: float):
        self.sigma2 = sigma2
        self.d = d
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, u: dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:
        return (
            0.5
            / self.sigma2
            * ufl.inner(u * ufl.exp(m) - self.d, u * ufl.exp(m) - self.d)
            * self.dx
        )


def run_inversion(
    mesh_filename: str,
    nx: int,
    ny: int,
    noise_variance: float,
    prior_param: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    sep = "\n" + "#" * 80 + "\n"
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    fname = mesh_filename
    fid = dlx.io.XDMFFile(comm, fname, "r")
    msh = fid.read_mesh(name="mesh")
    Vh_phi = dlx.fem.functionspace(msh, ("Lagrange", 2))
    Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]
    ndofs = [
        Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs,
        Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
    ]

    hpx.master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
    hpx.master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

    # FORWARD MODEL
    u0 = 1.0
    D = 1.0 / 24.0
    pde_handler = DiffusionApproximation(D, u0)

    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [], is_fwd_linear=True)
    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)
    m_true.interpolate(
        lambda x: np.log(0.01)
        + 3.0 * (((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0)) < 1.0),
    )
    m_true.x.scatter_forward()

    m_true = m_true.x
    u_true = pde.generate_state()

    x_true = [u_true, m_true, None]
    pde.solveFwd(u_true, x_true)
    xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]
    # LIKELIHOOD
    hpx.updateFromVector(xfun[hpx.STATE], u_true)
    u_fun_true = xfun[hpx.STATE]
    hpx.updateFromVector(xfun[hpx.PARAMETER], m_true)
    m_fun_true = xfun[hpx.PARAMETER]

    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun_true * ufl.exp(m_fun_true)
    hpx.projection(expr, d)
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance), d.x)
    d.x.scatter_forward()
    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form)
    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = np.log(0.01)
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(Vh_m, prior_param["gamma"], prior_param["delta"], mean=prior_mean)
    model = hpx.Model(pde, prior, misfit)

    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)
    hpx.parRandom.normal(1.0, noise)
    prior.sample(noise, m0)

    data_misfit_True = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=True, verbose=(rank == 0),
    )

    data_misfit_False = hpx.modelVerify(
        model, m0, is_quadratic=False, misfit_only=False, verbose=(rank == 0),
    )

    # # #######################################

    intial_guess_m = prior.generate_parameter(0)
    intial_guess_m.array[:] = prior_mean.array[:]
    x = [
        model.generate_vector(hpx.STATE),
        intial_guess_m,
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
    parameters["GN_iter"] = 20
    if rank != 0:
        parameters["print_level"] = -1

    solver = hpx.ReducedSpaceNewtonCG(model, parameters)

    x = solver.solve(x)

    if solver.converged:
        hpx.master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        hpx.master_print(comm, "\nNot Converged")
    hpx.master_print(comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    hpx.master_print(comm, "Final gradient norm: ", solver.final_grad_norm)
    hpx.master_print(comm, "Final cost: ", solver.final_cost)

    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER], name="m_map")
    m_true_fun = hpx.vector2Function(m_true, Vh[hpx.PARAMETER], name="m_true")

    V_P1 = dlx.fem.functionspace(msh, ("Lagrange", 1))

    u_true_fun = dlx.fem.Function(V_P1, name="u_true")
    u_true_fun.interpolate(hpx.vector2Function(u_true, Vh[hpx.STATE]))
    u_true_fun.x.scatter_forward()

    u_map_fun = dlx.fem.Function(V_P1, name="u_map")
    u_map_fun.interpolate(hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE]))
    u_map_fun.x.scatter_forward()

    d_fun = dlx.fem.Function(V_P1, name="data")
    d_fun.interpolate(d)
    d_fun.x.scatter_forward()

    with dlx.io.VTXWriter(
        msh.comm,
        "qpact_BiLaplacian_prior_np{0:d}_Prior.bp".format(nproc),
        [m_fun, m_true_fun, u_map_fun, u_true_fun, d_fun],
    ) as vtx:
        vtx.write(0.0)

    optimizer_results = {}
    if solver.termination_reasons[solver.reason] == "Norm of the gradient less than tolerance":
        optimizer_results["optimizer"] = True
    else:
        optimizer_results["optimizer"] = False

    Hmisfit = hpx.ReducedHessian(model, misfit_only=True)

    k = 80
    p = 20
    if rank == 0:
        print("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k, p))

    Omega = hpx.MultiVector(x[hpx.PARAMETER].petsc_vec, k + p)

    hpx.parRandom.normal(1.0, Omega)

    d, U = hpx.doublePassG(Hmisfit.mat, prior.R, prior.Rsolver, Omega, k, s=1)

    # generating prior and posterior samples
    lap_aprx = hpx.LaplaceApproximator(prior, d, U)
    lap_aprx.mean = prior.generate_parameter(0)
    lap_aprx.mean.array[:] = x[hpx.PARAMETER].array[:]

    noise = prior.generate_parameter("noise")

    num_samples_generate = 5
    use_vtx = False
    prior_sample = dlx.fem.Function(Vh[hpx.PARAMETER], name="prior_sample")
    posterior_sample = dlx.fem.Function(Vh[hpx.PARAMETER], name="posterior_sample")
    if use_vtx:
        with dlx.io.VTXWriter(
            msh.comm,
            "pact_prior_Bilaplacian_samples_np{0:d}.bp".format(nproc),
            [prior_sample, posterior_sample],
        ) as vtx:
            for i in range(num_samples_generate):
                hpx.parRandom.normal(1.0, noise)
                lap_aprx.sample(noise, prior_sample.x, posterior_sample.x)
                prior_sample.x.scatter_forward()
                posterior_sample.x.scatter_forward()
                vtx.write(float(i))
    else:
        ############################################
        with dlx.io.XDMFFile(
            msh.comm,
            "pact_prior_Bilaplacian_samples_np{0:d}.xdmf".format(nproc),
            "w",
        ) as file:
            file.write_mesh(msh)
            for i in range(num_samples_generate):
                hpx.parRandom.normal(1.0, noise)
                lap_aprx.sample(noise, prior_sample.x, posterior_sample.x)
                prior_sample.x.scatter_forward()
                posterior_sample.x.scatter_forward()
                file.write_function(prior_sample, float(i))
                file.write_function(posterior_sample, float(i))

    eigen_decomposition_results = {"A": Hmisfit, "B": prior, "k": k, "d": d, "U": U}

    final_results = {
        "data_misfit_True": data_misfit_True,
        "data_misfit_False": data_misfit_False,
        "optimizer_results": optimizer_results,
        "eigen_decomposition_results": eigen_decomposition_results,
    }

    return final_results
    ######################################


if __name__ == "__main__":
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.040, "delta": 0.8}
    mesh_filename = "./meshes/circle.xdmf"
    final_results = run_inversion(mesh_filename, nx, ny, noise_variance, prior_param)
    k, d = (
        final_results["eigen_decomposition_results"]["k"],
        final_results["eigen_decomposition_results"]["d"],
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        plt.savefig("qpact_result_FD_Gradient_Hessian_Check")
        plt.figure()
        plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
        plt.yscale("log")
        plt.savefig("qpact_Eigen_Decomposition_results.png")
