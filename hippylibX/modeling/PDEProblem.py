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

        self.Wuu = None
        self.Wmu = None
        self.Wmm = None
        self.A = None
        self.C = None
        
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0, "adjoint": 0, "incremental_forward": 0, "incremental_adjoint": 0}

    def generate_state(self) -> dlx.la.Vector:
        """ Return a vector in the shape of the state. """
        return dlx.la.vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        
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

            res_form = self.varf_handler(u, mfun, p)

            A_form = ufl.lhs(res_form)
            
            b_form = ufl.rhs(res_form)
        
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form), bcs=self.bc)
        
            A.assemble()
            self.solver.setOperators(A)
            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))
        
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
    

    def setLinearizationPoint(self,x : list, gauss_newton_approx) -> None:
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(3)]
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = ufl.derivative(f_form, x_fun[i])
            

        self.A = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ), self.bc0 )
        self.A.assemble()

        self.At = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative( g_form[STATE],x_fun[ADJOINT] )  ), self.bc0 )
        self.At.assemble()

        self.C = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), self.bc0)
        self.C.assemble()

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.setOperators(self.A)
        self.solver_adj_inc.setOperators(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wuu = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.derivative(g_form[STATE],x_fun[STATE])), self.bc0)
            self.Wuu.assemble()
            
            Wuu_t = self.Wuu.copy()
            Wuu_t.transpose()            

            # [bc.zero(Wuu_t) for bc in self.bc0] I am assuming this is not needed as self.Wuu 
            # was assembled incorporating the boundary conditions in self.bc0, so they would be 
            # reflected in Wuu_t
            
            # self.Wuu = Transpose(Wuu_t)
            self.Wuu = Wuu_t.copy()
            self.Wuu.transpose()


            self.Wmu = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])),self.bc0)
            self.Wmu.assemble()

            Wmu_t = self.Wmu.copy()
            Wmu_t.transpose()
            
            self.Wmu = Wmu_t.copy()
            self.Wmu.transpose()
            
            self.Wmm = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.derivative(g_form[PARAMETER],x_fun[PARAMETER])))
            self.Wmm.assemble()

    def apply_ij(self,i : int, j : int, dir : dlx.la.Vector, out : dlx.la.Vector) ->  None:   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm    
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i >= j:
            if KKT[i,j] is None:
                dlx.la.create_petsc_vector_wrap(out).scale(0.)
            else:
                KKT[i,j].mult(dlx.la.create_petsc_vector_wrap(dir), dlx.la.create_petsc_vector_wrap(out))
                
        else:
            if KKT[j,i] is None:
                dlx.la.create_petsc_vector_wrap(out).scale(0.)
            else:
                KKT[j,i].multTranspose(dlx.la.create_petsc_vector_wrap(dir), dlx.la.create_petsc_vector_wrap(out))
                
    

    def solveIncremental(self, out : dlx.la.Vector, rhs : dlx.la.Vector, is_adj : bool) -> None:        
        """ If :code:`is_adj == False`:

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
            self.solver_adj_inc.solve(dlx.la.create_petsc_vector_wrap(rhs), dlx.la.create_petsc_vector_wrap(out))
        
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(dlx.la.create_petsc_vector_wrap(rhs), dlx.la.create_petsc_vector_wrap(out))
    

