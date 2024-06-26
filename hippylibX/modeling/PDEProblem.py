# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import dolfinx as dlx
import ufl
import petsc4py
from .variables import STATE, PARAMETER, ADJOINT
from ..utils.vector2function import updateFromVector


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


class PDEVariationalProblem:
    def __init__(self, Vh: list, varf_handler, bc=[], bc0=[], is_fwd_linear=False):
        self.Vh = Vh
        self.varf_handler = varf_handler

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

        self.xfun_trial = [ufl.TrialFunction(Vhi) for Vhi in Vh]

        self.xfun_test = [ufl.TestFunction(Vhi) for Vhi in Vh]

        self.bc = bc
        self.bc0 = bc0

        self.Wuu = None
        self.Wmu = None
        self.Wum = None
        self.Wmm = None

        self.A = None
        self.At = None
        self.C = None

        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {
            "forward": 0,
            "adjoint": 0,
            "incremental_forward": 0,
            "incremental_adjoint": 0,
        }

        self.petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    def __del__(self):
        # self.solver.destroy()
        # self.solver_fwd_inc.destroy()
        # self.solver_adj_inc.destroy()

        if self.Wuu is not None:
            self.Wuu.destroy()

        if self.Wmu is not None:
            self.Wmu.destroy()

        if self.Wum is not None:
            self.Wum.destroy()

        if self.Wmm is not None:
            self.Wmm.destroy()

        if self.A is not None:
            self.A.destroy()

        if self.At is not None:
            self.At.destroy()

        if self.C is not None:
            self.C.destroy()

    def generate_state(self) -> dlx.la.Vector:
        """Return a vector in the shape of the state."""
        return dlx.la.vector(
            self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs
        )

    def generate_parameter(self) -> dlx.la.Vector:
        """Return a vector in the shape of the parameter."""
        return dlx.la.vector(
            self.Vh[PARAMETER].dofmap.index_map, self.Vh[PARAMETER].dofmap.index_map_bs
        )

    def solveFwd(self, state: dlx.la.Vector, x: list) -> None:
        """Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""

        updateFromVector(self.xfun[PARAMETER], x[PARAMETER])
        mfun = self.xfun[PARAMETER]

        self.n_calls["forward"] += 1

        if self.solver is None:
            self.solver = self._createLUSolver()
            self.solver.setTolerances(rtol=1e-9)

        # def monitor(ksp,its,rnorm):
        #     print(ksp.view())

        # self.solver.setMonitor(monitor)

        if self.is_fwd_linear:
            u = ufl.TrialFunction(self.Vh[STATE])
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, mfun, p)

            A_form = ufl.lhs(res_form)

            b_form = ufl.rhs(res_form)

            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form), bcs=self.bc)

            A.assemble()
            self.solver.setOperators(A)
            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))
            dlx.fem.petsc.apply_lifting(b, [dlx.fem.form(A_form)], [self.bc])
            b.ghostUpdate(
                petsc4py.PETSc.InsertMode.ADD_VALUES, petsc4py.PETSc.ScatterMode.REVERSE
            )
            dlx.fem.petsc.set_bc(b, self.bc)
            self.solver.solve(b, state.petsc_vec)

            A.destroy()
            b.destroy()

    def solveAdj(
        self, adj: dlx.la.Vector, x: dlx.la.Vector, adj_rhs: dlx.la.Vector
    ) -> None:
        """Solve the linear adjoint problem:
        Given :math:`m, u`; find :math:`p` such that
        .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """

        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        updateFromVector(self.xfun[STATE], x[STATE])
        u = self.xfun[STATE]

        updateFromVector(self.xfun[PARAMETER], x[PARAMETER])
        m = self.xfun[PARAMETER]

        p = dlx.fem.Function(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = ufl.derivative(ufl.derivative(varf, u, du), p, dp)

        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form), bcs=self.bc0)

        Aadj.assemble()

        self.solver.setOperators(Aadj)

        self.solver.solve(adj_rhs.petsc_vec, adj.petsc_vec)

        Aadj.destroy()

    def evalGradientParameter(self, x: list, out: dlx.la.Vector) -> None:
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.`"""

        updateFromVector(self.xfun[STATE], x[STATE])
        u = self.xfun[STATE]

        updateFromVector(self.xfun[PARAMETER], x[PARAMETER])
        m = self.xfun[PARAMETER]

        updateFromVector(self.xfun[ADJOINT], x[ADJOINT])
        p = self.xfun[ADJOINT]

        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)

        out.array[:] = 0.0

        dlx.fem.petsc.assemble_vector(
            out.petsc_vec, dlx.fem.form(ufl.derivative(res_form, m, dm))
        )
        out.petsc_vec.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES, petsc4py.PETSc.ScatterMode.REVERSE
        )

    def _createLUSolver(self) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create(self.Vh[0].mesh.comm)
        problem_prefix = f"dolfinx_solve_{id(self)}"
        ksp.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(problem_prefix)
        # petsc options for solver

        # Example:
        if self.petsc_options is not None:
            for k, v in self.petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()

        return ksp

    def setLinearizationPoint(self, x: list, gauss_newton_approx) -> None:
        """Set the values of the state and parameter
        for the incremental forward and adjoint solvers."""

        for i in range(3):
            updateFromVector(self.xfun[i], x[i])

        x_fun = self.xfun

        x_fun_trial = self.xfun_trial
        x_fun_test = self.xfun_test

        f_form = self.varf_handler(*x_fun)

        g_form = [None, None, None]

        for i in range(3):
            g_form[i] = ufl.derivative(f_form, x_fun[i], x_fun_test[i])

        if self.A is None:
            self.A = dlx.fem.petsc.create_matrix(
                dlx.fem.form(
                    ufl.derivative(g_form[ADJOINT], x_fun[STATE], x_fun_trial[STATE])
                )
            )

        self.A.zeroEntries()
        dlx.fem.petsc.assemble_matrix(
            self.A,
            dlx.fem.form(
                ufl.derivative(g_form[ADJOINT], x_fun[STATE], x_fun_trial[STATE])
            ),
            self.bc0,
        )
        self.A.assemble()

        if self.At is None:
            self.At = dlx.fem.petsc.create_matrix(
                dlx.fem.form(
                    ufl.derivative(g_form[STATE], x_fun[ADJOINT], x_fun_trial[ADJOINT])
                )
            )

        self.At.zeroEntries()
        dlx.fem.petsc.assemble_matrix(
            self.At,
            dlx.fem.form(
                ufl.derivative(g_form[STATE], x_fun[ADJOINT], x_fun_trial[ADJOINT])
            ),
            self.bc0,
        )
        self.At.assemble()

        if self.C is None:
            self.C = dlx.fem.petsc.create_matrix(
                dlx.fem.form(
                    ufl.derivative(
                        g_form[ADJOINT], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                    )
                )
            )

        self.C.zeroEntries()
        dlx.fem.petsc.assemble_matrix(
            self.C,
            dlx.fem.form(
                ufl.derivative(
                    g_form[ADJOINT], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                )
            ),
            bcs=self.bc0,
            diagonal=0.0,
        )
        self.C.assemble()

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

        self.solver_fwd_inc.setOperators(self.A)
        self.solver_adj_inc.setOperators(self.At)

        if gauss_newton_approx:
            if self.Wuu is not None:
                self.Wuu.destroy()
            self.Wuu = None

            if self.Wmu is not None:
                self.Wmu.destroy()
            self.Wmu = None

            if self.Wum is not None:
                self.Wum.destroy()
            self.Wum = None

            if self.Wmm is not None:
                self.Wmm.destroy()
            self.Wmm = None

        else:
            if self.Wuu is None:
                self.Wuu = dlx.fem.petsc.create_matrix(
                    dlx.fem.form(
                        ufl.derivative(g_form[STATE], x_fun[STATE], x_fun_trial[STATE])
                    )
                )

            self.Wuu.zeroEntries()
            dlx.fem.petsc.assemble_matrix(
                self.Wuu,
                dlx.fem.form(
                    ufl.derivative(g_form[STATE], x_fun[STATE], x_fun_trial[STATE])
                ),
                self.bc0,
                diagonal=0.0,
            )
            self.Wuu.assemble()

            if self.Wum is None:
                self.Wum = dlx.fem.petsc.create_matrix(
                    dlx.fem.form(
                        ufl.derivative(
                            g_form[STATE], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                        )
                    )
                )

            self.Wum.zeroEntries()
            dlx.fem.petsc.assemble_matrix(
                self.Wum,
                dlx.fem.form(
                    ufl.derivative(
                        g_form[STATE], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                    )
                ),
                self.bc0,
            )
            self.Wum.assemble()

            if self.Wmu is not None:
                self.Wmu.destroy()

            self.Wmu = self.Wum.copy()
            self.Wmu.transpose()

            if self.Wmm is None:
                self.Wmm = dlx.fem.petsc.create_matrix(
                    dlx.fem.form(
                        ufl.derivative(
                            g_form[PARAMETER], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                        )
                    )
                )

            self.Wmm.zeroEntries()
            dlx.fem.petsc.assemble_matrix(
                self.Wmm,
                dlx.fem.form(
                    ufl.derivative(
                        g_form[PARAMETER], x_fun[PARAMETER], x_fun_trial[PARAMETER]
                    )
                ),
            )
            self.Wmm.assemble()

    def apply_ij(self, i: int, j: int, dir: dlx.la.Vector, out: dlx.la.Vector) -> None:
        """
        Given :math:`u, m, p`; compute
        :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`disr`,
        :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE, STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C

        if i >= j:
            if KKT[i, j] is None:
                out.petsc_vec.scale(0.0)
            else:
                KKT[i, j].mult(dir.petsc_vec, out.petsc_vec)

        else:
            if KKT[j, i] is None:
                out.petsc_vec.scale(0.0)

            else:
                KKT[j, i].multTranspose(dir.petsc_vec, out.petsc_vec)

    def solveIncremental(
        self, out: dlx.la.Vector, rhs: dlx.la.Vector, is_adj: bool
    ) -> None:
        """If :code:`is_adj == False`:

        Solve the forward incremental system:
        Given :math:`u, m`, find :math:`\\tilde{u}` such that

            .. math:: \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.

        If :code:`is_adj == True`:

        Solve the adjoint incremental system:
        Given :math:`u, m`, find :math:`\\tilde{p}` such that

            .. math:: \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(rhs.petsc_vec, out.petsc_vec)

        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(rhs.petsc_vec, out.petsc_vec)
