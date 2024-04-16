# testing the lowRankHessian.mult method from modeling/laplaceApproximation.py
# x, y petsc vectors
# h1 = Hlr(x)
# h2 = Hlr(y)
# result1 = h2.h1 (inner product)

# h1 = Hlr(x)
# h2 = Hlr(h1)
# result2 = y.dot(h2) (inner product)

# result1 == result2

import unittest
import sys
import os
import numpy as np
import dolfinx as dlx

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

        Hlr = hpx.LowRankHessian(prior, d, U)
        vec1 = dlx.la.vector(prior.Vh.dofmap.index_map)
        hpx.parRandom.normal(1.0, vec1)

        vec2 = dlx.la.vector(prior.Vh.dofmap.index_map)
        hpx.parRandom.normal(1.0, vec2)

        temp_petsc_vec1 = dlx.la.create_petsc_vector_wrap(vec1)
        temp_petsc_vec2 = dlx.la.create_petsc_vector_wrap(vec2)

        help1 = dlx.la.create_petsc_vector(
            prior.Vh.dofmap.index_map, prior.Vh.dofmap.bs
        )
        help2 = dlx.la.create_petsc_vector(
            prior.Vh.dofmap.index_map, prior.Vh.dofmap.bs
        )

        Hlr.mult(temp_petsc_vec1, help1)
        Hlr.mult(temp_petsc_vec2, help2)

        result_1 = help2.dot(help1)

        Hlr.mult(temp_petsc_vec1, help1)
        Hlr.mult(help1, help2)
        result_2 = temp_petsc_vec2.dot(help2)

        temp_petsc_vec1.destroy()
        temp_petsc_vec2.destroy()
        help1.destroy()
        help2.destroy()

        self.assertLessEqual(
            np.abs(result_1 - result_2),
            1e-3,
            "lowRankHessian.mult failed",
        )


if __name__ == "__main__":
    unittest.main()
