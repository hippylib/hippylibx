from mpi4py import MPI
import dolfinx as dlx
import numpy as np
from petsc4py import PETSc
import unittest

import sys
import os

sys.path.append(os.path.abspath("../.."))
import hippylibX as hpx
from hippylibX.modeling.pointwiseInterpolationMatrix import findPoints


def gather_global(y, comm):
    """
    Gather the PETSc vector y = P u on every process.

    The global numbering of y is the order of x (the row local-to-global map of
    P only affects insertion with local indices), so gathering by global index
    returns the values in the order of x.
    """
    scatter, y0 = PETSc.Scatter.toZero(y)
    scatter(y, y0, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    y_global = comm.bcast(y0.array.copy() if comm.rank == 0 else None, root=0)
    scatter.destroy()
    y0.destroy()
    return y_global


class Testing_Execution(unittest.TestCase):
    def test2d_scalar_CG1(self):
        # Mesh and function space
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1))

        # Interpolation points
        x = np.array(
            [
                [0.2, 0.3],
                [0.7, 0.4],
                [0.9, 0.8],
            ]
        )

        # Build interpolation matrix
        P = hpx.pointwiseInterpolationMatrix(V, x)

        # Example function
        u = dlx.fem.Function(V)
        u.interpolate(lambda x: x[0] + 2 * x[1])

        # Apply interpolation matrix
        u_vec = u.x.petsc_vec

        y = P.createVecLeft()
        P.mult(u_vec, y)

        # ------------------------------------------------------------
        # Gather interpolated values on rank 0
        # ------------------------------------------------------------
        comm = MPI.COMM_WORLD
        rank = comm.rank

        y_global = gather_global(y, comm)

        if rank == 0:
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

    def test2d_vector_CG1(self):
        # Mesh and vector function space
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1, (2,)))

        # Interpolation points
        x = np.array(
            [
                [0.2, 0.3],
                [0.7, 0.4],
                [0.9, 0.8],
            ],
            dtype=np.float64,
        )

        # Build interpolation matrix
        P = hpx.pointwiseInterpolationMatrix(V, x)

        # Example vector-valued function
        #
        # u(x,y) = [x + 2y,
        #           3x - y]
        #
        u = dlx.fem.Function(V)

        u.interpolate(lambda x: np.vstack((x[0] + 2.0 * x[1], 3.0 * x[0] - x[1])))

        # Apply interpolation matrix
        u_vec = u.x.petsc_vec

        y = P.createVecLeft()
        P.mult(u_vec, y)

        # ------------------------------------------------------------
        # Gather interpolated values on rank 0
        # ------------------------------------------------------------
        comm = MPI.COMM_WORLD
        rank = comm.rank

        y_global = gather_global(y, comm).reshape(len(x), 2)

        if rank == 0:
            # True values
            y_true = np.column_stack((x[:, 0] + 2.0 * x[:, 1], 3.0 * x[:, 0] - x[:, 1]))

            print("Interpolated values:")
            print(y_global)

            print("True values:")
            print(y_true)

            print("Error:")
            print(np.abs(y_global - y_true))

            np.testing.assert_allclose(
                y_global,
                y_true,
                rtol=1e-12,
                atol=1e-12,
            )

    def _check_dense_grid(self, domain, V, f, ngrid):
        """
        Interpolate on a dense grid that includes points on mesh vertices,
        facets, the boundary and (in parallel) the interfaces between
        processes, plus a few points outside the mesh.
        """
        comm = domain.comm
        gdim = domain.geometry.dim
        g = np.linspace(0.0, 1.0, ngrid)
        x = np.stack([c.ravel() for c in np.meshgrid(*([g] * gdim), indexing="ij")], axis=1)
        x_out = np.full((2, gdim), 1.5)
        x_out[1, 0] = -0.5
        x = np.vstack([x, x_out])
        npoints = x.shape[0]
        n_inside = npoints - x_out.shape[0]

        P = hpx.pointwiseInterpolationMatrix(V, x)
        bs_val = 1 if len(V.element.value_shape) == 0 else int(np.prod(V.element.value_shape))
        self.assertEqual(P.getSize()[0], npoints * bs_val)

        # Points computed on this process
        x3 = np.zeros((npoints, 3))
        x3[:, :gdim] = x
        _, cells, point_ids = findPoints(domain, x3)
        self.assertEqual(P.getLocalSize()[0], len(point_ids) * bs_val)

        # Every point is computed by exactly one process
        ids_all = np.concatenate(comm.allgather(point_ids))
        np.testing.assert_array_equal(np.sort(ids_all), np.arange(npoints))

        # ... in a cell it owns (or on rank 0 if outside the mesh)
        num_owned_cells = domain.topology.index_map(domain.topology.dim).size_local
        inside = point_ids < n_inside
        self.assertTrue(np.all((cells[inside] >= 0) & (cells[inside] < num_owned_cells)))
        if np.any(~inside):
            self.assertEqual(comm.rank, 0)
            self.assertTrue(np.all(cells[~inside] == -1))

        # Row local-to-global map: local row block j -> global row block point_ids[j]
        row_lgmap = P.getLGMap()[0]
        nloc = len(point_ids) * bs_val
        expected_rows = (point_ids[:, None] * bs_val + np.arange(bs_val)).ravel()
        np.testing.assert_array_equal(
            row_lgmap.apply(np.arange(nloc, dtype=np.int32)), expected_rows
        )

        # Row sums (global rows in the order of x): 1 inside the mesh, 0 outside
        ones = P.createVecRight()
        ones.set(1.0)
        rs = P.createVecLeft()
        P.mult(ones, rs)
        expected = np.zeros(npoints * bs_val)
        expected[: n_inside * bs_val] = 1.0
        np.testing.assert_allclose(gather_global(rs, comm), expected, rtol=0, atol=1e-12)

        # Exact interpolation of a function in V, same ordering as in serial
        u = dlx.fem.Function(V)
        u.interpolate(f)
        y = P.createVecLeft()
        P.mult(u.x.petsc_vec, y)
        y_global = gather_global(y, comm)
        y_true = np.asarray(f(x.T)).T.reshape(npoints, -1)
        y_true[n_inside:] = 0.0
        np.testing.assert_allclose(y_global, y_true.ravel(), rtol=0, atol=1e-12)

    def test2d_dense_grid_scalar_CG1(self):
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1))
        self._check_dense_grid(domain, V, lambda x: x[0] + 2 * x[1], 41)

    def test2d_dense_grid_scalar_CG2(self):
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
        V = dlx.fem.functionspace(domain, ("Lagrange", 2))
        self._check_dense_grid(domain, V, lambda x: x[0] ** 2 - x[0] * x[1], 25)

    def test2d_dense_grid_vector_CG1(self):
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1, (3,)))
        self._check_dense_grid(
            domain,
            V,
            lambda x: np.vstack((x[0] + 2.0 * x[1], 3.0 * x[0] - x[1], 1.0 + 0 * x[0])),
            41,
        )

    def test3d_dense_grid_scalar_CG1(self):
        domain = dlx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
        V = dlx.fem.functionspace(domain, ("Lagrange", 1))
        self._check_dense_grid(domain, V, lambda x: x[0] - x[1] + 3 * x[2], 9)

    def test2d_RT(self):
        # Mesh and vector function space
        domain = dlx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = dlx.fem.functionspace(domain, ("RT", 2))

        # Interpolation points
        x = np.array(
            [
                [0.2, 0.3],
                [0.7, 0.4],
                [0.9, 0.8],
            ],
            dtype=np.float64,
        )

        with self.assertRaises(ValueError):
            # Build interpolation matrix
            _ = hpx.pointwiseInterpolationMatrix(V, x)


if __name__ == "__main__":
    unittest.main()
