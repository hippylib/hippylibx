############################################################
import dolfinx as dlx
import ufl
import petsc4py
from .variables import STATE, PARAMETER, ADJOINT
#from ..algorithms.linalg import Transpose #not yet used
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function


#decorator for functions in classes that are not used -> may not be needed in the final
#version of X
def unused_function(func):
    return None


class PDEVariationalProblem:
    def __init__(self, Vh : dlx.fem.FunctionSpace, varf_handler, bc=[], bc0=[], is_fwd_linear=False):
        self.Vh = Vh
        self.varf_handler = varf_handler

        self.bc = bc        
        self.bc0 = bc0

        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0, "adjoint": 0, "incremental_forward": 0, "incremental_adjoint": 0}

    def generate_state(self) -> dlx.la.Vector:
        """ Return a vector in the shape of the state. """
        return dlx.la.vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        
    # @unused_function #now being used in mode.generate_vector() in modelVerify.py
    def generate_parameter(self) -> dlx.la.Vector:
        """ Return a vector in the shape of the parameter. """
        return dlx.la.vector(self.Vh[PARAMETER].dofmap.index_map, self.Vh[PARAMETER].dofmap.index_map_bs) 
        

    @unused_function   
    def init_parameter(self, m):
        """ Initialize the parameter."""
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.dofmap.index_map)

    def solveFwd(self, state : dlx.la.Vector, x : dlx.la.Vector) -> None: #state is a vector
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""

        mfun = vector2Function(x[PARAMETER],self.Vh[PARAMETER])

        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        if self.is_fwd_linear:    
            u = ufl.TrialFunction(self.Vh[STATE])   
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, mfun, p) #all 3 arguments-dl.Function types

            A_form = ufl.lhs(res_form) #ufl.form.Form
            
            b_form = ufl.rhs(res_form)

            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form), bcs=self.bc) #petsc4py.PETSc.Mat
        
            A.assemble() #petsc4py.PETSc.Mat
            self.solver.setOperators(A)
            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form)) #petsc4py.PETSc.Vec
        
            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

            state_vec = dlx.la.create_petsc_vector_wrap(state)
            self.solver.solve(b,state_vec)                     

    def solveAdj(self, adj : dlx.la.Vector, x : dlx.la.Vector, adj_rhs : petsc4py.PETSc.Vec ) -> None: 

        """ Solve the linear adjoint problem:
        Given :math:`m, u`; find :math:`p` such that
        .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """

        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dlx.fem.Function(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = ufl.derivative( ufl.derivative(varf, u, du), p, dp )
        
        
        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs = self.bc0)
            
        Aadj.assemble()

        self.solver.setOperators(Aadj)

        #Also works
        self.solver.solve(adj_rhs,dlx.la.create_petsc_vector_wrap(adj) )
        
    def evalGradientParameter(self, x : list, out : dlx.la.Vector) -> None:
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)

        tmp = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, dm))) 
        tmp.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        dlx.la.create_petsc_vector_wrap(out).scale(0.)
        dlx.la.create_petsc_vector_wrap(out).axpy(1.,tmp)


    def _createLUSolver(self) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create()
        return ksp
    
    @unused_function
    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(3)]
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        [bc.zero(self.C) for bc in self.bc0]
                
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            [bc.zero(self.Wuu) for bc in self.bc0]
            Wuu_t = Transpose(self.Wuu)
            [bc.zero(Wuu_t) for bc in self.bc0]
            self.Wuu = Transpose(Wuu_t)
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            Wmu_t = Transpose(self.Wmu)
            [bc.zero(Wmu_t) for bc in self.bc0]
            self.Wmu = Transpose(Wmu_t)
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]))