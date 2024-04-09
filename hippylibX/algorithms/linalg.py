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
        init_vector=None,
    ) -> None:
        self.S = S

        if hasattr(self.S, "generate_vector"):
            self.tmp = self.S.generate_vector()
        elif hasattr(self.S, "getOperators"):
            M, _ = self.S.getOperators()
            self.tmp = M.createVecLeft()
        else:
            raise NotImplementedError(
                "one of generate vector or getOperators not implemented for Solver object"
            )

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.S.solve(x, y)
