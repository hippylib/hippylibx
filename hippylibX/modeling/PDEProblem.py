############################################################
import dolfinx as dlx
import ufl
import petsc4py
from .variables import STATE, PARAMETER, ADJOINT
#from ..algorithms.linalg import Transpose #not yet used
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function, updateFromVector


#decorator for functions in classes that are not used -> may not be needed in the final
#version of X
def unused_function(func):
    return None


class PDEVariationalProblem:
    def __init__(self, Vh : list, varf_handler, bc=[], bc0=[], is_fwd_linear=False):
        self.Vh = Vh
        self.varf_handler = varf_handler

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

        self.bc = bc        
        self.bc0 = bc0

        self.Wuu = None
        self.Wmu = None
        self.Wmm = None
        self.A = None
        self.At = None
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

    def solveFwd(self, state : dlx.la.Vector, x : list) -> None: #state is a vector
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""

        # mfun = vector2Function(x[PARAMETER],self.Vh[PARAMETER])

        updateFromVector(self.xfun[PARAMETER], x[PARAMETER])
        mfun = self.xfun[PARAMETER]


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
            state_vec.destroy()
            A.destroy()
            b.destroy()

    def solveAdj(self, adj : dlx.la.Vector, x : dlx.la.Vector, adj_rhs : petsc4py.PETSc.Vec ) -> None: 

        """ Solve the linear adjoint problem:
        Given :math:`m, u`; find :math:`p` such that
        .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """

        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        # u = vector2Function(x[STATE], self.Vh[STATE])
        # m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        
        updateFromVector(self.xfun[STATE],x[STATE])
        u = self.xfun[STATE]

        updateFromVector(self.xfun[PARAMETER],x[PARAMETER])
        m = self.xfun[PARAMETER]
                
        p = dlx.fem.Function(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = ufl.derivative( ufl.derivative(varf, u, du), p, dp )
                
        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs = self.bc0)
            
        Aadj.assemble()

        self.solver.setOperators(Aadj)

        temp_petsc_vec_adj = dlx.la.create_petsc_vector_wrap(adj)
        # self.solver.solve(adj_rhs,dlx.la.create_petsc_vector_wrap(adj) )
        self.solver.solve(adj_rhs, temp_petsc_vec_adj )
        temp_petsc_vec_adj.destroy()
        Aadj.destroy()

    def evalGradientParameter(self, x : list, out : dlx.la.Vector) -> None:
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        
        # u = vector2Function(x[STATE], self.Vh[STATE])
        # m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        # p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])

        updateFromVector(self.xfun[STATE],x[STATE])
        u = self.xfun[STATE]

        updateFromVector(self.xfun[PARAMETER],x[PARAMETER])
        m = self.xfun[PARAMETER]

        updateFromVector(self.xfun[ADJOINT],x[ADJOINT])
        p = self.xfun[ADJOINT]
        
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)

        tmp = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, dm))) 
        tmp.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        
        # dlx.la.create_petsc_vector_wrap(out).scale(0.)
        # dlx.la.create_petsc_vector_wrap(out).axpy(1.,tmp)
        
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)
        
        temp_petsc_vec_out.scale(0.)
        temp_petsc_vec_out.axpy(1., tmp)
        
        temp_petsc_vec_out.destroy()
        tmp.destroy()
        

    def _createLUSolver(self) -> petsc4py.PETSc.KSP:
        ksp = petsc4py.PETSc.KSP().create()
        return ksp
    

    def setLinearizationPoint(self,x : list, gauss_newton_approx) -> None:
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        # x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(3)]

        for i in range(3):
            updateFromVector(self.xfun[i],x[i])

        x_fun = self.xfun

        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None]
        
        for i in range(3):
            g_form[i] = ufl.derivative(f_form, x_fun[i])
            
        
        # if self.A is None:
        #     self.A = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ), self.bc0 )
        # else:
        #     self.A.zeroEntries()
        #     dlx.fem.petsc.assemble_matrix(self.A, dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ), self.bc0 )
        # self.A.assemble()

        # if self.At is None:
        #     self.At = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative( g_form[STATE],x_fun[ADJOINT] )  ), self.bc0 )
        # else:
        #     self.At.zeroEntries()
        #     dlx.fem.petsc.assemble_matrix(self.At, dlx.fem.form( ufl.derivative( g_form[STATE],x_fun[ADJOINT] )  ), self.bc0 )
        
        # self.At.assemble()


        # if self.C is None:
        #     self.C = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), self.bc0)
        # else:
        #     self.C.zeroEntries()
        #     dlx.fem.petsc.assemble_matrix(self.C, dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), self.bc0)
        
        # self.C.assemble()
    
        #######################################################
        # if self.A is None:
        #     self.A = dlx.fem.petsc.create_matrix(dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ) )

        # self.A.zeroEntries()
        # dlx.fem.petsc.assemble_matrix(self.A, dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ), self.bc0 )
        # self.A.assemble()

        # self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0)

        if self.A is None:
            self.A = dlx.fem.petsc.create_matrix(dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ) )

        self.A.zeroEntries()
        dlx.fem.petsc.assemble_matrix(self.A, dlx.fem.form( ufl.derivative( g_form[ADJOINT],x_fun[STATE] )  ), self.bc0 )
        self.A.assemble()

        # print(self.A.getDiagonal().array[:].min(),":",self.A.getDiagonal().array[:].max())


        # self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)

        if self.At is None:
            self.At = dlx.fem.petsc.create_matrix(dlx.fem.form( ufl.derivative( g_form[STATE],x_fun[ADJOINT] )  ) )
        
        self.At.zeroEntries()
        dlx.fem.petsc.assemble_matrix(self.At, dlx.fem.form( ufl.derivative( g_form[STATE],x_fun[ADJOINT] )  ), self.bc0 )
        self.At.assemble()

        # self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        # [bc.zero(self.C) for bc in self.bc0]    

        # if self.C is None:
        #     self.C = dlx.fem.petsc.create_matrix(dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])))

        # self.C.zeroEntries() #gives error, if before self.C.assemble() - [0] Not for matrices where you have set values but not yet assembled

        # dlx.fem.petsc.assemble_matrix(self.C, dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), bcs = self.bc0, diagonal = 0.)        
        # dlx.fem.petsc.assemble_matrix(self.C, dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), bcs = self.bc0)        
        # dlx.fem.petsc.assemble_matrix(self.C, dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])), diagonal = 0.)        
        # dlx.fem.petsc.assemble_matrix(self.C, dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])))        

        self.C = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.derivative(g_form[ADJOINT],x_fun[PARAMETER])),bcs = self.bc0, diagonal = 0.)
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
            # self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            # [bc.zero(self.Wuu) for bc in self.bc0]
            # Wuu_t = Transpose(self.Wuu)
            # [bc.zero(Wuu_t) for bc in self.bc0]
            # self.Wuu = Transpose(Wuu_t)
            # self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            # Wmu_t = Transpose(self.Wmu)
            # [bc.zero(Wmu_t) for bc in self.bc0]
            # self.Wmu = Transpose(Wmu_t)
            # self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER
            
            # if self.Wuu is None:
            #     self.Wuu = dlx.fem.petsc.create_matrix(dlx.fem.form(ufl.derivative(g_form[STATE],x_fun[STATE])))

            # self.Wuu.zeroEntries()
            # dlx.fem.petsc.assemble_matrix(self.Wuu, dlx.fem.form(ufl.derivative(g_form[STATE],x_fun[STATE])))
            # self.Wuu.assemble()

            if self.Wuu is None:
                self.Wuu = dlx.fem.petsc.create_matrix(dlx.fem.form(ufl.derivative(g_form[STATE],x_fun[STATE])))
            
            self.Wuu.zeroEntries()
            dlx.fem.petsc.assemble_matrix(self.Wuu, dlx.fem.form(ufl.derivative(g_form[STATE],x_fun[STATE])), self.bc0, diagonal = 0.)
            self.Wuu.assemble()
            
            # print(self.Wuu.getDiagonal().array[:].min(),":",self.Wuu.getDiagonal().array[:].max())

            Wuu_t = self.Wuu.copy()
            Wuu_t.transpose()   

            # print(Wuu_t.getDiagonal().array[:])
            # print(Wuu_t.getDiagonal().array[:].min(),":",Wuu_t.getDiagonal().array[:].max())

            # temp_diag_vec = Wuu_t.getDiagonal()
            # temp_diag_vec.scale(0.)         
            # Wuu_t.setDiagonal(temp_diag_vec)
            # temp_diag_vec.destroy()         


            #don't know how to do this: [bc.zero(Wuu_t) for bc in self.bc0]


            # [bc.zero(Wuu_t) for bc in self.bc0] I am assuming this is not needed as self.Wuu 
            # was assembled incorporating the boundary conditions in self.bc0, so they would be 
            # reflected in Wuu_t
            
            # self.Wuu = Transpose(Wuu_t)
    
            self.Wuu = Wuu_t.copy()
            self.Wuu.transpose()

            # if self.Wmu is None:
            #     self.Wmu = dlx.fem.petsc.create_matrix(dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))

            # self.Wmu.zeroEntries() #[0] Not for matrices where you have set values but not yet assembled
            # dlx.fem.petsc.assemble_matrix(self.Wmu, dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))
            
            # if self.Wmu is None:
            #     self.Wmu = dlx.fem.petsc.create_matrix(dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))
            
            # self.Wmu.zeroEntries()
            # dlx.fem.petsc.assemble_matrix(self.Wmu, dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))
            # self.Wmu.assemble()

            # test_val = dlx.fem.petsc.assemble_vector(dlx.fem.form(g_form[PARAMETER]))
            # print(test_val.array[:].min(),":",test_val.array[:].max())

            # print(x_fun[STATE].x.array[:].min(),":",x_fun[STATE].x.array[:].max())

            self.Wmu = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))
            self.Wmu.assemble()
 
            # print(self.Wmu.getDiagonal().array[:])
            # print(self.Wmu.getDiagonal().array[:].min(),":",self.Wmu.getDiagonal().array[:].max())


            # if self.Wmu is None:
            #     self.Wmu = dlx.fem.petsc.assemble_matrix(dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))

            # # self.Wmu.zeroEntries() #[0] Not for matrices where you have set values but not yet assembled
            # # dlx.fem.petsc.assemble_matrix(self.Wmu, dlx.fem.form( ufl.derivative(g_form[PARAMETER],x_fun[STATE])))
            # self.Wmu.assemble()

            Wmu_t = self.Wmu.copy()
            Wmu_t.transpose()


            # temp_diag_vec = Wmu_t.getDiagonal()
            # temp_diag_vec.scale(0.)         
            # Wmu_t.setDiagonal(temp_diag_vec)
            # temp_diag_vec.destroy()         

            # print(Wmu_t.getDiagonal().array[:].min(),":",Wmu_t.getDiagonal().array[:].max())

            #Don't know how to do this: [bc.zero(Wmu_t) for bc in self.bc0]

            self.Wmu = Wmu_t.copy()
            self.Wmu.transpose()

            if self.Wmm is None:
                self.Wmm = dlx.fem.petsc.create_matrix(dlx.fem.form(ufl.derivative(g_form[PARAMETER],x_fun[PARAMETER])))

            self.Wmm.zeroEntries()
            dlx.fem.petsc.assemble_matrix(self.Wmm, dlx.fem.form(ufl.derivative(g_form[PARAMETER],x_fun[PARAMETER])))
            self.Wmm.assemble()

            # print(self.Wmm.getDiagonal().array[:].min(),":",self.Wmm.getDiagonal().array[:].max())



    def apply_ij(self,i : int, j : int, dir : dlx.la.Vector, out : dlx.la.Vector) ->  None:   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`disr`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm    
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)
        temp_petsc_vec_dir = dlx.la.create_petsc_vector_wrap(dir)
        

        if i >= j:
            if KKT[i,j] is None:
                # dlx.la.create_petsc_vector_wrap(out).scale(0.)
                temp_petsc_vec_out.scale(0.)
            else:
                # KKT[i,j].mult(dlx.la.create_petsc_vector_wrap(dir), dlx.la.create_petsc_vector_wrap(out))
                KKT[i,j].mult(temp_petsc_vec_dir, temp_petsc_vec_out)
                
        else:
            if KKT[j,i] is None:
                # dlx.la.create_petsc_vector_wrap(out).scale(0.)
                temp_petsc_vec_out.scale(0.)
            else:
                # KKT[j,i].multTranspose(dlx.la.create_petsc_vector_wrap(dir), dlx.la.create_petsc_vector_wrap(out))
                KKT[j,i].multTranspose(temp_petsc_vec_dir, temp_petsc_vec_out)
        
        temp_petsc_vec_out.destroy()
        temp_petsc_vec_dir.destroy()
                    
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
        temp_petsc_vec_rhs = dlx.la.create_petsc_vector_wrap(rhs)
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)
        
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            # self.solver_adj_inc.solve(dlx.la.create_petsc_vector_wrap(rhs), dlx.la.create_petsc_vector_wrap(out))
            self.solver_adj_inc.solve(temp_petsc_vec_rhs, temp_petsc_vec_out)
        
        else:
            self.n_calls["incremental_forward"] += 1
            # self.solver_fwd_inc.solve(dlx.la.create_petsc_vector_wrap(rhs), dlx.la.create_petsc_vector_wrap(out))
            self.solver_fwd_inc.solve(temp_petsc_vec_rhs, temp_petsc_vec_out)
    
        temp_petsc_vec_out.destroy()
        temp_petsc_vec_rhs.destroy()
        

