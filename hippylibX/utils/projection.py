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
from petsc4py import PETSc


class L2Projector:
    """Assemble and apply the L2 projector for a finite element space."""

    def __init__(self, V, bcs=None):
        self.V = V
        self.bcs = [] if bcs is None else list(bcs)
        self.comm = self.V.mesh.comm
        self.dx = ufl.dx(domain=self.V.mesh)

        w = ufl.TestFunction(self.V)
        Pv = ufl.TrialFunction(self.V)

        self.a = dlx.fem.form(ufl.inner(Pv, w) * self.dx)
        self.A = dlx.fem.petsc.assemble_matrix(self.a, self.bcs)
        self.A.assemble()

        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setFromOptions()

    def project(self, v, target_func):
        """Project ``v`` onto the space of ``target_func`` in-place."""
        L = dlx.fem.form(ufl.inner(v, w := ufl.TestFunction(self.V)) * self.dx)
        b = dlx.fem.petsc.assemble_vector(L)

        try:
            dlx.fem.petsc.apply_lifting(b, [self.a], [self.bcs])
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            dlx.fem.petsc.set_bc(b, self.bcs)
            self.solver.solve(b, target_func.x.petsc_vec)
            target_func.x.scatter_forward()
        finally:
            b.destroy()

    def __del__(self):
        if hasattr(self, "solver"):
            self.solver.destroy()
        if hasattr(self, "A"):
            self.A.destroy()


def projection(v, target_func, bcs=None):
    """
    Project expression ``v`` onto the function space of ``target_func``.

    The result is written in-place into ``target_func``.
    """
    projector = L2Projector(target_func.function_space, bcs)
    projector.project(v, target_func)
