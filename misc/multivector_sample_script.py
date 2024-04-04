# script to perform B-Orthogonalization of a multivector - Omega and return the result
# Omega.dot(Bq) which is expected to be an identity matrix of the order Omega.nvec.

import sys
import os
from mpi4py import MPI
import dolfinx as dlx
import dolfinx.fem.petsc
import ufl

sys.path.append(os.path.abspath("../"))
import hippylibX as hpx


def multi_vector_testing(nx, ny, nvec):
    comm = MPI.COMM_WORLD
    msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
    Vh = dlx.fem.FunctionSpace(msh, ("Lagrange", 1))
    trial = ufl.TrialFunction(Vh)
    test = ufl.TestFunction(Vh)
    varfM = ufl.inner(trial, test) * ufl.Measure(
        "dx", metadata={"quadrature_degree": 4}
    )
    M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
    M.assemble()
    sample_vec1 = dlx.fem.Function(Vh)
    sample_vec1 = sample_vec1.x
    sample_petsc_vec = dlx.la.create_petsc_vector_wrap(sample_vec1)
    Omega = hpx.MultiVector.createFromVec(sample_petsc_vec, nvec)
    sample_petsc_vec.destroy()
    hpx.parRandom.normal(1.0, Omega)
    Bq, _ = Omega.Borthogonalize(M)
    result = Omega.dot(Bq)
    return result


if __name__ == "__main__":
    nx, ny, nvec = 10, 10, 3
    result = multi_vector_testing(nx, ny, nvec)
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(result)
