from mpi4py import MPI
import dolfinx as dlx
import numpy as np
import unittest

import sys
import os

sys.path.append(os.path.abspath("../.."))
import hippylibX as hpx


class Testing_Execution(unittest.TestCase):
    def test2d_scalar_CG1(self):
        # Mesh and function space
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1))
    
        # Interpolation points
        x = np.array([
            [0.2, 0.3],
            [0.7, 0.4],
            [0.9, 0.8],
        ])

        # Build interpolation matrix
        P = hpx.pointwiseInterpolationMatrix(V, x)

        # Example function
        u = dlx.fem.Function(V)
        u.interpolate(lambda x: x[0] + 2*x[1])

        # Apply interpolation matrix
        u_vec = u.x.petsc_vec

        y = P.createVecLeft()
        P.mult(u_vec, y)

        # ------------------------------------------------------------
        # Gather interpolated values on rank 0
        # ------------------------------------------------------------
        comm = MPI.COMM_WORLD
        rank = comm.rank

        y_local = y.array

        gathered = comm.gather(y_local, root=0)

        if rank == 0:
            # Concatenate contributions from all ranks
            y_global = np.concatenate(gathered)

            # True values
            y_true = x[:, 0] + 2.0 * x[:, 1]

            print("Interpolated values:", y_global)
            print("True values:        ", y_true)
            print("Error:              ", np.abs(y_global - y_true))

            np.testing.assert_allclose(
                y_global,
                y_true,
                rtol=1e-12,
                atol=1e-12,
            )

if __name__ == "__main__":
    unittest.main()
