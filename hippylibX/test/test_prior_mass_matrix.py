# x -> random dolfinx Vector
# in modeling/prior.py:
# sqrtM . sqrtM.T . x == M . x

import dolfinx as dlx
from mpi4py import MPI
import sys
import os
import dolfinx.fem.petsc
import petsc4py
import unittest
import numpy as np

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx


class Testing_Execution(unittest.TestCase):
    def test_prior_mass_matrix(self):
        nx, ny = 10, 10
        prior_param = {"gamma": 0.03, "delta": 0.3}
        comm = MPI.COMM_WORLD
        msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
        Vh = dlx.fem.functionspace(msh, ("Lagrange", 1))

        prior = hpx.BiLaplacianPrior(
            Vh, prior_param["gamma"], prior_param["delta"], mean=None
        )

        x = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        hpx.parRandom.replay()
        hpx.parRandom.normal(1.0, x)
        y1 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        y2 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        tmp = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)

        prior.sqrtM.multTranspose(x.petsc_vec, tmp.petsc_vec)
        prior.sqrtM.mult(tmp.petsc_vec, y1.petsc_vec)

        prior.M.mult(x.petsc_vec, y2.petsc_vec)

        y2.petsc_vec.axpy(-1.0, y1.petsc_vec)

        value = y2.petsc_vec.norm(petsc4py.PETSc.NormType.N2)
        self.assertLessEqual(
            np.abs(value),
            1e-6,
            "prior_sqrtM creation failed",
        )



if __name__ == "__main__":
    unittest.main()
