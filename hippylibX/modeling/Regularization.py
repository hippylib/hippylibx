import dolfinx as dlx
import ufl
import petsc4py
from mpi4py import MPI
import hippylibX as hpx


# functional handler for prior:
class H1TikhonvFunctional:
    def __init__(self, gamma, delta, m0=None):
        self.gamma = gamma  # These are dlx Constant, Expression, or Function
        self.delta = delta
        self.m0 = m0
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, m):  # Here m is a dlx Function
        if self.m0 is None:
            dm = m
        else:
            dm = m - self.m0

        return (
            ufl.inner(self.gamma * ufl.grad(dm), ufl.grad(dm)) * self.dx
            + ufl.inner(self.delta * dm, dm) * self.dx
        )


class VariationalRegularization:
    def __init__(
        self, Vh: dlx.fem.FunctionSpace, functional_handler, isQuadratic=False
    ):
        self.Vh = Vh
        self.functional_handler = functional_handler  # a function or a functor that takes as input m (as dlx.Function) and evaluates the regularization functional
        self.isQuadratic = isQuadratic  # Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian depends on m
        self.mfun = dlx.fem.Function(Vh)
        self.mtest = ufl.TestFunction(Vh)
        self.mtrial = ufl.TrialFunction(Vh)
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

        self.petsc_options_M = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }
        self.petsc_options_R = {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": "1e-12",
            "ksp_max_it": "1000",
            "ksp_error_if_not_converged": "true",
            "ksp_initial_guess_nonzero": "false",
        }

        self.R = None
        self.Rsolver = None

        if self.isQuadratic:
            tmp = dlx.fem.Function(Vh).x
            self.setLinearizationPoint(tmp)

        self.M = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(self.mtrial, self.mtest) * self.dx)
        )
        self.M.assemble()

        self.Msolver = self._createsolver(self.petsc_options_M)
        self.Msolver.setOperators(self.M)

    def __del__(self):
        if self.Rsolver is not None:
            self.Rsolver.destroy()
        if self.R is not None:
            self.R.destroy()
        self.Msolver.destroy()
        self.M.destroy()

    def _createsolver(self, petsc_options) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create(self.Vh.mesh.comm)
        problem_prefix = f"dolfinx_solve_{id(self)}"
        ksp.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = petsc4py.PETSc.Options()
        opts.prefixPush(problem_prefix)

        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        ksp.setFromOptions()

        return ksp

    def cost(self, m: dlx.la.Vector):
        hpx.updateFromVector(self.mfun, m)
        cost_functional = self.functional_handler(self.mfun)
        local_cost = dlx.fem.assemble_scalar(dlx.fem.form(cost_functional))
        return self.Vh.mesh.comm.allreduce(local_cost, op=MPI.SUM)

    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> dlx.la.Vector:
        hpx.updateFromVector(self.mfun, m)
        out.array[:] = 0.0
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        dlx.fem.petsc.assemble_vector(
            tmp_out,
            dlx.fem.form(
                ufl.derivative(
                    self.functional_handler(self.mfun), self.mfun, self.mtest
                )
            ),
        )
        tmp_out.ghostUpdate(
            petsc4py.PETSc.InsertMode.ADD_VALUES, petsc4py.PETSc.ScatterMode.REVERSE
        )
        tmp_out.destroy()

    def setLinearizationPoint(self, m: dlx.la.Vector, gauss_newton_approx=False):
        if self.isQuadratic and (self.R is not None):
            return
        hpx.updateFromVector(self.mfun, m)
        L = ufl.derivative(
            ufl.derivative(self.functional_handler(self.mfun), self.mfun, self.mtest),
            self.mfun,
            self.mtrial,
        )
        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(L))
        self.R.assemble()
        self.Rsolver = self._createsolver(self.petsc_options_R)
        self.Rsolver.setOperators(self.R)
