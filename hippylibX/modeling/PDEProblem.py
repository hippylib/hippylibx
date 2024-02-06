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
    def __init__(self, Vh, varf_handler, bc=[], bc0=[], is_fwd_linear=False):
        self.Vh = Vh
        self.varf_handler = varf_handler

        self.bc = bc        
        self.bc0 = bc0

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0, "adjoint": 0, "incremental_forward": 0, "incremental_adjoint": 0}

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        # return dlx.fem.Function(self.Vh[STATE]).vector
        # return dlx.la.create_petsc_vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        
        return dlx.la.vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        
    # @unused_function #now being used in mode.generate_vector() in modelVerify.py
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        # return dlx.la.create_petsc_vector(self.Vh[PARAMETER].dofmap.index_map, self.Vh[PARAMETER].dofmap.index_map_bs) 
        return dlx.la.vector(self.Vh[PARAMETER].dofmap.index_map, self.Vh[PARAMETER].dofmap.index_map_bs) 
        

    @unused_function   
    def init_parameter(self, m):
        """ Initialize the parameter."""
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.dofmap.index_map)

    def solveFwd(self, state, x): #state is a vector
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""

        mfun = vector2Function(x[PARAMETER],self.Vh[PARAMETER])

        # print("first:")
        # print(x[PARAMETER].min())
        # print(x[PARAMETER].max())

        # print(x[PARAMETER],'\n')
        # temp_vec = dlx.la.create_petsc_vector(self.Vh[PARAMETER].dofmap.index_map,self.Vh[PARAMETER].dofmap.index_map_bs) 
        # temp_vec.x.array[:] = x[PARAMETER]

        # temp_m_vec = x[PARAMETER].copy()
        # mfun = vector2Function(temp_m_vec,self.Vh[PARAMETER])

        #once the function scope ends, the values in the underlying vector also disappear.
        #As per: https://fenicsproject.discourse.group/t/manipulating-vector-data-of-fem-function/11056/2
        #Need to find a way to work around this.
        #Ans: return the function? 


        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        if self.is_fwd_linear:    
            u = ufl.TrialFunction(self.Vh[STATE])   
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, mfun, p) #all 3 arguments-dl.Function types

            # print("second:")
            # print(x[PARAMETER].min())
            # print(x[PARAMETER].max())
        
            A_form = ufl.lhs(res_form) #ufl.form.Form
            
            # print("third:")
            # print(x[PARAMETER].min())
            # print(x[PARAMETER].max()) #garbage

            b_form = ufl.rhs(res_form)
            
            # print("fourth:")
            # print(x[PARAMETER].min()) 
            # print(x[PARAMETER].max()) #garbage
        
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form), bcs=self.bc) #petsc4py.PETSc.Mat
            
            # print("fifth:")
            # print(x[PARAMETER].min()) #garbage
            # print(x[PARAMETER].max()) #garbage
        
            A.assemble() #petsc4py.PETSc.Mat
            self.solver.setOperators(A)
            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form)) #petsc4py.PETSc.Vec
        
            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            # dlx.fem.petsc.set_bc(b,self.bc)
            
            # self.solver.solve(b,state)
            # print(type(state))
            state_vec = dlx.la.create_petsc_vector_wrap(state)
            self.solver.solve(b,state_vec)
            
            state_vec.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            state_vec.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
                    
            # print(state.array.min(),":",state.array.max())
    
            # return A
         

    # pde.solveAdj(adj_vec, x_true, adj_rhs)
    def solveAdj(self, adj, x, adj_rhs, comm):

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
        
        # print(adj_rhs.vector.min(),":",adj_rhs.vector.max()) #-1.0788096613719298, 1.9211903386280702
        
        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs = self.bc0)
        
        # print(adj_rhs.min(),":",adj_rhs.max()) #-3.420763640362111e+306, 1.1652105010162572e+301

        Aadj.assemble()

        self.solver.setOperators(Aadj)

        #not needed:        
        # dlx.fem.petsc.apply_lifting(adj_rhs.vector,[dlx.fem.form(adj_form)],[self.bc0])            
        # adj_rhs.vector.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        # adj_rhs.vector.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
        
        # adj_rhs = create_petsc_vector_wrap(adj_rhs.x,comm)
        adj_vec = dlx.la.create_petsc_vector_wrap(adj)

        # self.solver.solve(adj_rhs, adj)
        self.solver.solve(adj_rhs, adj_vec)

        adj_vec.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        adj_vec.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
            

        # self.solver.solve(adj_rhs, adj)


        ####################################
        # u = vector2Function(x[STATE], self.Vh[STATE])
        # m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        # p = dl.Function(self.Vh[ADJOINT])
        # du = dl.TestFunction(self.Vh[STATE])
        # dp = dl.TrialFunction(self.Vh[ADJOINT])
        # varf = self.varf_handler(u, m, p)
        # adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        # Aadj, dummy = dl.assemble_system(adj_form, ufl.inner(u,du)*ufl.dx, self.bc0)
        # self.solver.set_operator(Aadj)
        # self.solver.solve(adj, adj_rhs)
        ####################################





        # print(adj_rhs.min(),":",adj_rhs.max())

        ######################################
        # p = dlx.fem.Function(self.Vh[ADJOINT])
        # du = ufl.TestFunction(self.Vh[STATE])
        # dp = ufl.TrialFunction(self.Vh[ADJOINT])
        # x_state_fun = vector2Function(x[STATE],self.Vh[STATE])
        # varf = self.varf_handler(x_state_fun, vector2Function(x[PARAMETER],self.Vh[PARAMETER]), p) 
        # adj_form = ufl.derivative(ufl.derivative(varf, x_state_fun, du), p, dp)

        # Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs=self.bc0)
        # Aadj.assemble()
        # self.solver.setOperators(Aadj)

        # #b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form)) #petsc4py.PETSc.Vec

        # dlx.fem.petsc.apply_lifting(adj_rhs,[dlx.fem.form(adj_form)],[self.bc0])
        # adj_rhs.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

        # self.solver.solve(adj_rhs,adj)
        # ######################################
        
    # self.problem.evalGradientParameter(x, mg)
        
    def evalGradientParameter(self, x):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)

        eval_grad = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, dm)))
        eval_grad.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

        return eval_grad


    def _createLUSolver(self):
        ksp = petsc4py.PETSc.KSP().create()
        return ksp


    def solveAdj_2(self, adj, x, adj_rhs):

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
        
        # print(adj_rhs.vector.min(),":",adj_rhs.vector.max()) #-1.0788096613719298, 1.9211903386280702
        
        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs = self.bc0)
        
        # print(adj_rhs.vector.min(),":",adj_rhs.vector.max()) #-3.420763640362111e+306, 1.1652105010162572e+301

        Aadj.assemble()

        self.solver.setOperators(Aadj)

        #not needed:        
        # dlx.fem.petsc.apply_lifting(adj_rhs.vector,[dlx.fem.form(adj_form)],[self.bc0])            
        # adj_rhs.vector.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        # adj_rhs.vector.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
                    
        self.solver.solve(adj_rhs.vector, adj)
