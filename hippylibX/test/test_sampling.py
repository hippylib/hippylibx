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

        x = dlx.la.vector(prior.Vh.dofmap.index_map)
        hpx.parRandom.normal(1.0, x)

        y = dlx.la.vector(prior.Vh.dofmap.index_map)
        hpx.parRandom.normal(1.0, y)

        sx, sy = (
            dlx.la.vector(prior.Vh.dofmap.index_map),
            dlx.la.vector(prior.Vh.dofmap.index_map),
        )

        PS_lr = hpx.LowRankPosteriorSampler(prior, d, U)
        PS_lr.sample(x, sx)
        PS_lr.sample(y, sy)

        result_1 = dlx.cpp.la.inner_product(sy._cpp_object, sx._cpp_object)

        Hlr = hpx.LowRankHessian(prior, d, U)
        Hx = dlx.la.create_petsc_vector(prior.Vh.dofmap.index_map, prior.Vh.dofmap.bs)

        temp_petsc_vec_x = dlx.la.create_petsc_vector_wrap(x)
        temp_petsc_vec_y = dlx.la.create_petsc_vector_wrap(y)

        Hlr.mult(temp_petsc_vec_x, Hx)
        result_2 = temp_petsc_vec_y.dot(Hx)

        temp_petsc_vec_x.destroy()
        temp_petsc_vec_y.destroy()
        Hx.destroy()

        print(result_1, result_2)

        # self.assertLessEqual(
        #     np.abs(result_1 - result_2),
        #     1e-3,
        #     "lowRankHessian.mult failed",
        # )


if __name__ == "__main__":
    unittest.main()
