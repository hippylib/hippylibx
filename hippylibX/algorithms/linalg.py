import dolfinx as dlx
from mpi4py import MPI
import petsc4py


def inner(x: dlx.la.Vector, y: dlx.la.Vector):
    return dlx.cpp.la.inner_product(x._cpp_object, y._cpp_object)


class Solver2Operator:
    def __init__(self, S, mpi_comm=MPI.COMM_WORLD, init_vector=None):
        self.S = S
        # self.tmp = dlx.la.Vector(mpi_comm)
        self.tmp = self.S.createVecLeft()

        # self.my_init_vector = init_vector

    #     if self.my_init_vector is None:
    #         if hasattr(self.S, "init_vector"):
    #             self.my_init_vector = self.S.init_vector
    #         elif hasattr(self.S, "operator"):
    #             self.my_init_vector = self.S.operator().init_vector
    #         elif hasattr(self.S, "get_operator"):
    #             self.my_init_vector = self.S.get_operator().init_vector

    # def init_vector(self, x, dim):
    #     if self.my_init_vector:
    #         self.my_init_vector(x,dim)
    #     else:
    #         raise NotImplementedError("Solver2Operator.init_vector")

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec):
        # self.S.solve(y,x) #reverse in Hippylib compared to petsc4py KSP
        self.S.solve(x, y)

    def inner(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec):
        # self.S.solve(self.tmp,y)
        self.S.solve(y, self.tmp)
        # return self.tmp.inner(x)
        return self.tmp.dot(x)
