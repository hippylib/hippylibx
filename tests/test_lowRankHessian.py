# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

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
import hippylibX as hpx

sys.path.append(os.path.abspath("../../example"))
from example import poisson_dirichlet_example


def low_Rank_Hessian_mult_solve(prior: Any, d: np.array, U: hpx.MultiVector) -> tuple[float, float]:
    Hlr = hpx.LowRankHessian(prior, d, U)
    vec1 = dlx.la.vector(prior.Vh.dofmap.index_map)
    vec2 = dlx.la.vector(prior.Vh.dofmap.index_map)
    vec3 = dlx.la.vector(prior.Vh.dofmap.index_map)

    hpx.parRandom.normal(1.0, vec1)

    Hlr.mult(vec1.petsc_vec, vec2.petsc_vec)
    Hlr.solve(vec2.petsc_vec, vec3.petsc_vec)
    vec3.petsc_vec.axpy(-1.0, vec1.petsc_vec)

    value1 = vec3.petsc_vec.norm(petsc4py.PETSc.NormType.N2)

    hpx.parRandom.normal(1.0, vec1)

    Hlr.solve(vec1.petsc_vec, vec2.petsc_vec)
    Hlr.mult(vec2.petsc_vec, vec3.petsc_vec)
    vec3.petsc_vec.axpy(-1.0, vec1.petsc_vec)

    value2 = vec3.petsc_vec.norm(petsc4py.PETSc.NormType.N2)

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
        out = poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, prior_param)
        prior, d, U = (
            out["eigen_decomposition_results"]["B"],
            out["eigen_decomposition_results"]["d"],
            out["eigen_decomposition_results"]["U"],
        )
        result = low_Rank_Hessian_mult_solve(prior, d, U)
        check_output(self, result)


if __name__ == "__main__":
    unittest.main()
