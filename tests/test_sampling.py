# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# testing the lowRankPosteriorSampler.sample method from modeling/laplaceApproximation.py
# x : random vector
# prior.Rsolver.solve(x, x1)
# Hlrsampler.sample(x1, x2)
# Hlrsampler.sample(x2, x3)
# Hlr.solve(x, y)
# assert x3 == y

import os
import sys
import unittest

import petsc4py

import dolfinx as dlx
import dolfinx.fem.petsc
import numpy as np

import hippylibX as hpx

sys.path.append(os.path.abspath("../example"))
import poisson_dirichlet_example


class Testing_Execution(unittest.TestCase):
    def test_sampling(self):
        hpx.parRandom.replay()

        nx = 64
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.0}
        out = poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, prior_param)
        prior, d, U = (
            out["eigen_decomposition_results"]["B"],
            out["eigen_decomposition_results"]["d"],
            out["eigen_decomposition_results"]["U"],
        )

        x = dlx.la.vector(prior.Vh.dofmap.index_map)
        x1 = dlx.la.vector(prior.Vh.dofmap.index_map)
        x2 = dlx.la.vector(prior.Vh.dofmap.index_map)
        x3 = dlx.la.vector(prior.Vh.dofmap.index_map)
        y = dlx.la.vector(prior.Vh.dofmap.index_map)

        hpx.parRandom.normal(1.0, x)

        prior.Rsolver.solve(x.petsc_vec, x1.petsc_vec)

        PS_lr = hpx.LowRankPosteriorSampler(prior, d, U)
        PS_lr.mult(x1, x2)
        PS_lr.mult(x2, x3)

        Hlr = hpx.LowRankHessian(prior, d, U)

        Hlr.solve(x.petsc_vec, y.petsc_vec)

        # assert x3 == y
        y.petsc_vec.axpy(-1.0, x3.petsc_vec)

        value = y.petsc_vec.norm(petsc4py.PETSc.NormType.N2)

        self.assertLessEqual(
            np.abs(value),
            1e-6,
            "lowRankPosteriorSampler.sample failed",
        )


if __name__ == "__main__":
    unittest.main()
