# Test to check if Hlr^(-1).Hlr == Hlr.Hlr(-1) == Identity
# Hlr -> hpx.LowRankHessian

# x ,y z as petsc4py Vectors:
# y = Hlr(x)
# z = Hlr^(-1)y => z == x

# y = Hlr(-1)x
# z = Hlr(y) => z == x

import unittest
import sys
import os
import numpy as np
import dolfinx as dlx
import dolfinx.fem.petsc
import petsc4py
from typing import Any

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx

sys.path.append(os.path.abspath("../../example"))
from example import poisson_dirichlet_example


def low_Rank_Hessian_mult_solve(
    prior: Any, d: np.array, U: hpx.MultiVector
) -> tuple[float, float]:
    Hlr = hpx.LowRankHessian(prior, d, U)
    vec1 = dlx.la.vector(prior.Vh.dofmap.index_map)
    vec2 = dlx.la.vector(prior.Vh.dofmap.index_map)
    vec3 = dlx.la.vector(prior.Vh.dofmap.index_map)

    hpx.parRandom.normal(1.0, vec1)
    temp_petsc_vec1 = dlx.la.create_petsc_vector_wrap(vec1)
    temp_petsc_vec2 = dlx.la.create_petsc_vector_wrap(vec2)
    temp_petsc_vec3 = dlx.la.create_petsc_vector_wrap(vec3)

    Hlr.mult(temp_petsc_vec1, temp_petsc_vec2)
    Hlr.solve(temp_petsc_vec2, temp_petsc_vec3)
    temp_petsc_vec3.axpy(-1.0, temp_petsc_vec1)

    value1 = temp_petsc_vec3.norm(petsc4py.PETSc.NormType.N2)

    hpx.parRandom.normal(1.0, vec1)

    Hlr.solve(temp_petsc_vec1, temp_petsc_vec2)
    Hlr.mult(temp_petsc_vec2, temp_petsc_vec3)
    temp_petsc_vec3.axpy(-1.0, temp_petsc_vec1)

    value2 = temp_petsc_vec3.norm(petsc4py.PETSc.NormType.N2)

    temp_petsc_vec1.destroy()
    temp_petsc_vec2.destroy()
    temp_petsc_vec3.destroy()

    return value1, value2


def check_output(self, result: tuple[float, float]):
    self.assertLessEqual(
        np.abs(result[0]),
        1e-6,
        "Hlr(-1).Hlr not equal to identity",
    )

    self.assertLessEqual(
        np.abs(result[1]),
        1e-6,
        "Hlr.Hlr(-1) not equal to identity",
    )


class Testing_Execution(unittest.TestCase):
    def test_lowRankHessian(self):
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
        result = low_Rank_Hessian_mult_solve(prior, d, U)
        check_output(self, result)


if __name__ == "__main__":
    unittest.main()
