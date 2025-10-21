# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# Low rank Hessian should be a very good preconditioner for the data
# misfit Hessian.

import os
import sys
import unittest

import dolfinx as dlx

import hippylibX as hpx

sys.path.append(os.path.abspath("../../example"))
from example import poisson_dirichlet_example


class Testing_Execution(unittest.TestCase):
    def test_lowRankHessian_precondtioner(self):
        hpx.parRandom.replay()
        nx = 64
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.0}
        out = poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, prior_param)
        Hmisfit, prior, d, U = (
            out["eigen_decomposition_results"]["A"],
            out["eigen_decomposition_results"]["B"],
            out["eigen_decomposition_results"]["d"],
            out["eigen_decomposition_results"]["U"],
        )
        lowRank_Hessian = hpx.LowRankHessian(prior, d, U)

        Hmisfit.misfit_only = False
        matr = Hmisfit.mat

        vec1 = dlx.la.vector(prior.Vh.dofmap.index_map)
        hpx.parRandom.normal(1.0, vec1)

        vec2 = dlx.la.vector(prior.Vh.dofmap.index_map)
        parameters = hpx.algorithms.cgsolverSteihaug.CGSolverSteihaug_ParameterList()
        solver = hpx.algorithms.cgsolverSteihaug.CGSolverSteihaug(parameters, prior.Vh.mesh.comm)
        solver.set_operator(matr)

        solver.set_preconditioner(prior.Rsolver)
        solver.solve(vec1, vec2)
        Rsolver_iter_count = solver.iter

        solver.set_preconditioner(lowRank_Hessian)
        solver.solve(vec1, vec2)
        Hlr_iter_count = solver.iter

        self.assertLess(
            Hlr_iter_count,
            Rsolver_iter_count,
            "lowRank Hessian not better pc than prior.Rsolver",
        )


if __name__ == "__main__":
    unittest.main()
