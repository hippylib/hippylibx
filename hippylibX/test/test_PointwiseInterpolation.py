from mpi4py import MPI
from dolfinx import mesh, fem
import numpy as np

# Mesh and function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
V = fem.functionspace(domain, ("Lagrange", 1))

# Interpolation points
x = np.array([
    [0.2, 0.3],
    [0.7, 0.4],
    [0.9, 0.8],
])

# Build interpolation matrix
P = point_interpolation_matrix(V, x)

# Example function
u = fem.Function(V)
u.interpolate(lambda x: x[0] + 2*x[1])

# Apply interpolation matrix
u_vec = u.x.petsc_vec

y = P.createVecLeft()
P.mult(u_vec, y)

print(y.array)
