import dolfinx as dlx
from mpi4py import MPI
import petsc4py
from typing import Any


def inner(x: dlx.la.Vector, y: dlx.la.Vector) -> float:
    return dlx.cpp.la.inner_product(x._cpp_object, y._cpp_object)


class Solver2Operator:
    def __init__(
        self,
        S: Any,
        mpi_comm=MPI.COMM_WORLD,
        # init_vector=None,
        createVecLeft = None,
        createVecRight = None
    ) -> None:
        self.S = S
        self.createVecLeft = createVecLeft
        self.createVecRight = createVecRight
         
    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.S.solve(x, y)
