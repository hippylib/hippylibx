# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

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
        createVecLeft=None,
        createVecRight=None,
    ) -> None:
        self.S = S
        self.createVecLeft = createVecLeft
        self.createVecRight = createVecRight

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.S.solve(x, y)
