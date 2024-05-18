# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import dolfinx as dlx
import ufl  # type: ignore
import petsc4py


def projection(v, target_func, bcs=[]):
    """
    Return projection of given expression :code:`v` onto the finite element
    space of a function :code:`target_func`.

    reference:
    https://github.com/michalhabera/dolfiny/blob/master/dolfiny/projection.py

    Inputs:
        - :code:`v`:  expression to project
        - :code:`target_func` : function that contains the projection
    """
    V = target_func.function_space
    dx = ufl.dx(V.mesh)
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dlx.fem.form(ufl.inner(Pv, w) * dx)
    L = dlx.fem.form(ufl.inner(v, w) * dx)

    bcs = []
    A = dlx.fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = dlx.fem.petsc.assemble_vector(L)
    dlx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
    )

    solver = petsc4py.PETSc.KSP().create()
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
