# testing the lowRankPosteriorSampler.sample method from modeling/laplaceApproximation.py
# x : random vector
# prior.Rsolver.solve(x, x1)
# Hlrsampler.sample(x1, x2)
# Hlrsampler.sample(x2, x3)
# Hlr.solve(x, y)
# assert x3 == y

import unittest
import sys
import os
import dolfinx as dlx
import petsc4py
import dolfinx.fem.petsc
import numpy as np

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx

sys.path.append(os.path.abspath("../../example"))
from example import poisson_dirichlet_example


class Testing_Execution(unittest.TestCase):
    def test_sampling(self):
        hpx.parRandom.replay()

        nx = 64
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.0}
        out = poisson_dirichlet_example.run_inversion(
            nx, ny, noise_variance, prior_param
        )
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

        temp_petsc_vec_x = dlx.la.create_petsc_vector_wrap(x)
        temp_petsc_vec_x1 = dlx.la.create_petsc_vector_wrap(x1)
        prior.Rsolver.solve(temp_petsc_vec_x, temp_petsc_vec_x1)

        PS_lr = hpx.LowRankPosteriorSampler(prior, d, U)
        PS_lr.mult(x1, x2)
        PS_lr.mult(x2, x3)

        Hlr = hpx.LowRankHessian(prior, d, U)

        temp_petsc_vec_y = dlx.la.create_petsc_vector_wrap(y)
        Hlr.solve(temp_petsc_vec_x, temp_petsc_vec_y)

        # assert x3 == y
        temp_petsc_vec_x3 = dlx.la.create_petsc_vector_wrap(x3)
        temp_petsc_vec_y.axpy(-1.0, temp_petsc_vec_x3)

        value = temp_petsc_vec_y.norm(petsc4py.PETSc.NormType.N2)

        temp_petsc_vec_x.destroy()
        temp_petsc_vec_x1.destroy()
        temp_petsc_vec_x3.destroy()
        temp_petsc_vec_y.destroy()

        self.assertLessEqual(
            np.abs(value),
            1e-6,
            "lowRankPosteriorSampler.sample failed",
        )


if __name__ == "__main__":
    unittest.main()
