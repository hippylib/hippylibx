# script to perform B-Orthogonalization of a multivector - Omega and return the result
# Omega.dot(Bq) which is expected to be an identity matrix of the order Omega.nvec.

import unittest
import sys
import os
import numpy as np
import dolfinx as dlx
from mpi4py import MPI
import ufl
import dolfinx.fem.petsc

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx


def multi_vector_testing(nx: int, ny: int, nvec: int) -> np.array:
    comm = MPI.COMM_WORLD
    msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
    Vh = dlx.fem.functionspace(msh, ("Lagrange", 1))
    trial = ufl.TrialFunction(Vh)
    test = ufl.TestFunction(Vh)
    varfM = ufl.inner(trial, test) * ufl.Measure(
        "dx", metadata={"quadrature_degree": 4}
    )
    M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
    M.assemble()
    sample_petsc_vec = dlx.fem.petsc.create_vector(dlx.fem.form(test * ufl.dx))
    Omega = hpx.MultiVector.createFromVec(sample_petsc_vec, nvec)
    sample_petsc_vec.destroy()
    hpx.parRandom.normal(1.0, Omega)
    Bq, _ = Omega.Borthogonalize(M)
    result = Omega.dot(Bq)
    return result


def check_output(self, result: np.array):
    self.assertTrue(result.shape[0] == result.shape[1])
    self.assertTrue(np.allclose(result, np.eye(result.shape[0]), atol=1e-6))


class Testing_Execution(unittest.TestCase):
    def test_multivector(self):
        hpx.parRandom.replay()
        nx = 64
        ny = 64
        nvec = 10
        result = multi_vector_testing(nx, ny, nvec)
        check_output(self, result)


if __name__ == "__main__":
    unittest.main()
