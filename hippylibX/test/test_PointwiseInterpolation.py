from mpi4py import MPI
import dolfinx as dlx
import numpy as np
import unittest

import sys
import os

sys.path.append(os.path.abspath("../.."))
import hippylibX as hpx


class Testing_Execution(unittest.TestCase):
    def test(self):
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
        P = hpx.point_interpolation_matrix(V, x)

        # Example function
        u = dlx.fem.Function(V)
        u.interpolate(lambda x: x[0] + 2*x[1])

        # Apply interpolation matrix
        u_vec = u.x.petsc_vec

        y = P.createVecLeft()
        P.mult(u_vec, y)

if __name__ == "__main__":
    unittest.main()

    print(y.array)
