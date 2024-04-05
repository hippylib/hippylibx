import dolfinx as dlx
from mpi4py import MPI
import petsc4py


def inner(x: dlx.la.Vector, y: dlx.la.Vector):
    return dlx.cpp.la.inner_product(x._cpp_object, y._cpp_object)


class Solver2Operator:
    def __init__(self, S, mpi_comm=MPI.COMM_WORLD, init_vector=None):
        self.S = S
        if isinstance(self.S, petsc4py.PETSc.KSP):
            M, _ = self.S.getOperators()
            self.tmp = M.createVecLeft()
        else:
            self.tmp = self.S.generate_vector()

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec):
        self.S.solve(x, y)

    def inner(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec):
        self.S.solve(y, self.tmp)
        return self.tmp.dot(x)
