import dolfinx as dlx
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl
import numpy as np
import petsc4py
# import numbers

#variables from 
# https://github.com/hippylib/hippylib/blob/35e4a3638f4b5eba926af65100b49f2ad51e22b0/hippylib/modeling/variables.py

STATE= 0
PARAMETER = 1
ADJOINT = 2
NVAR = 3

#decorator for functions in classes that are not used -> may not be needed in the final
#version of X

def unused_function(func):
    return None

def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """

    fun = dlx.fem.Function(Vh,**kwargs)
    fun.vector.axpy(1., x)
    fun.x.scatter_forward()
    return fun


@unused_function
def Transpose(A):
    A.assemble()
    AT = petsc4py.PETSc.Mat()
    AT = A.transpose()
    rmap,cmap = A.getLGMap()
    AT.setLGMap(cmap,rmap)
    return AT 

def _PETScLUSolver_set_operator(self, A):
    if hasattr(A, 'mat'):
        self.ksp().setOperators(A.mat())
    else:
        # self.ksp().setOperators(dl.as_backend_type(A).mat())
        self.ksp().setOperators(A.instance()) #dont know how to test this?


def PETScLUSolver(comm, method='default'):
    if not hasattr(petsc4py.PETSc.PETScLUSolver, 'set_operator'):
        petsc4py.PETSC.PETScLUSolver.set_operator = _PETScLUSolver_set_operator
    return petsc4py.PETSc.PETScLUSolver(comm, method)


class DiffusionApproximation:
    def __init__(self, D, u0, ds):
        """
        Define the forward model for the diffusion approximation to radiative transfer equations
        
        D: diffusion coefficient 1/mu_eff with mu_eff = sqrt(3 mu_a (mu_a + mu_ps) ), where mu_a
           is the unknown absorption coefficient, and mu_ps is the reduced scattering coefficient
           
        u0: Incident fluence (Robin condition)
        
        ds: boundary integrator for Robin condition
        """
        self.D = D
        self.u0 = u0
        self.ds = ds

        
    def __call__(self, u, m, p):
        return ufl.inner(self.D*ufl.grad(u), ufl.grad(p))*ufl.dx + \
            ufl.exp(m)*ufl.inner(u,p)*ufl.dx + \
            .5*ufl.inner(u-self.u0,p)*self.ds


class NonGaussianContinuousMisfit(object):
    def __init__(self, mesh, Vh, form):
        self.mesh = mesh
        self.Vh = Vh
        self.form = form

        self.x_lin_fun = None
        self.x_test = [ufl.TestFunction(Vh[STATE]), ufl.TestFunction(Vh[PARAMETER])]
        self.gauss_newton_approx = False

    def cost(self,x):
        loc_cost = dlx.fem.assemble_scalar(dlx.fem.form(self.form(vector2Function(x[STATE],Vh[STATE]), vector2Function(x[PARAMETER],Vh[PARAMETER]))))
        return self.mesh.comm.allreduce(loc_cost,op=MPI.SUM)

    def grad(self, i, x):
        x_state_fun,x_par_fun = vector2Function(x[STATE],Vh[STATE]), vector2Function(x[PARAMETER],Vh[PARAMETER]) 
        x_fun = [x_state_fun,x_par_fun] 
        loc_grad = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i])) ) #<class 'PETSc.Vec'>
        loc_grad.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        return loc_grad
    
      
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        u_fun = vector2Function(x[STATE], self.Vh[STATE])
        m_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx 
        

    #same issue as grad function, what is "type" of out that has to be passed as argument?
    def apply_ij(self,i,j, dir):
        form = self.form(*self.x_lin_fun)
        dir_fun = vector2Function(dir, self.Vh[j])
        action = ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun )
        loc_action = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(action) ) #<class 'PETSc.Vec'>
        loc_action.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        return loc_action

    
        
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
        #return function instead of vector, solveFwd using the function.vector object
        return dlx.la.create_petsc_vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 

    @unused_function
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dlx.la.create_petsc_vector(self.Vh[PARAMETER].dofmap.index_map, self.Vh[PARAMETER].dofmap.index_map_bs) 
   

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

        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        if self.is_fwd_linear:    
            u = ufl.TrialFunction(self.Vh[STATE])   
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, mfun, p) #all 3 arguments-dl.Function types
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form),bcs=self.bc)
            A.assemble() #petsc4py.PETSc.Mat

            self.solver.setOperators(A)

            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))

            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            # dlx.fem.petsc.set_bc(b,self.bc)

            self.solver.solve(b,state)

         
    def solveAdj(self, adj, x, adj_rhs):
        
        """ Solve the linear adjoint problem:
        Given :math:`m, u`; find :math:`p` such that
        .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        p = dlx.fem.Function(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        x_state_fun = vector2Function(x[STATE],self.Vh[STATE])
        varf = self.varf_handler(x_state_fun, vector2Function(x[PARAMETER],self.Vh[PARAMETER]), p) 
        adj_form = ufl.derivative(ufl.derivative(varf, x_state_fun, du), p, dp)

        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs=self.bc0)
        Aadj.assemble()
        self.solver.setOperators(Aadj)
        self.solver.solve(adj_rhs,adj)
   

    def evalGradientParameter(self, x):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)

        return dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, dm)))

        # dlx.fem.assemble( ufl.derivative(res_form, m, dm), tensor=out)


    def _createLUSolver(self):
        # return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())   
        #need to find substitute for below
        ksp = petsc4py.PETSc.KSP().create()
        # pc = ksp.getPC()
        # ksp.setFromOptions()
        return ksp

    

def master_print(comm, *args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)

class PACTMisfitForm:
    def __init__(self, d, sigma2):
        self.sigma2 = sigma2
        self.d = d
        
    def __call__(self,u,m):        
        return .5/self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx


##########working#############
nx = 64
ny = 64

noise_variance = 1e-6
prior_param = {"gamma": 0.05, "delta": 1.}
sep = "\n"+"#"*80+"\n"    

comm = MPI.COMM_WORLD
rank  = comm.rank
nproc = comm.size

fname = 'meshes/circle.xdmf'
fid = dlx.io.XDMFFile(comm,fname,"r")
msh = fid.read_mesh(name='mesh')


Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
Vh = [Vh_phi, Vh_m, Vh_phi]

# FORWARD MODEL
u0 = 1.
D = 1./24.
pde_handler = DiffusionApproximation(D, u0, ufl.ds)     #returns a ufl form
pde = PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)


# GROUND TRUTH
m_true = dlx.fem.Function(Vh_m)
m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
m_true.x.scatter_forward()
m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>
u_true = pde.generate_state()   #a vector, not a function, <class 'petsc4py.PETSc.Vec'>
x_true = [u_true, m_true, None]     #list of petsc vectors
pde.solveFwd(u_true, x_true) #petsc4py.PETSc.Vec


#LIKELIHOOD
u_fun_true = vector2Function(u_true, Vh[STATE]) 
m_fun_true = vector2Function(m_true, Vh[PARAMETER])





#Method 1: to create projection
#taken from: https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_elasticity.py#L213-L232

test_expr = dlx.fem.Expression(  u_fun_true*ufl.exp(m_fun_true), Vh[STATE].element.interpolation_points()  )
d = dlx.fem.Function(Vh[STATE])
d.interpolate(test_expr)
d.x.scatter_forward()


#Method 2: Taken from: https://github.com/michalhabera/dolfiny/blob/master/dolfiny/projection.py

# v.interpolate(  u_fun_true*ufl.exp(m_fun_true) ,Vh[STATE].element.interpolation_points())
# v = u_fun_true*ufl.exp(m_fun_true)
# d = dlx.fem.Function(Vh[STATE])
# V = d.function_space
# dx = ufl.dx(V.mesh)

# w = ufl.TestFunction(V)
# Pv = ufl.TrialFunction(V)
# a = dlx.fem.form(ufl.inner(Pv, w) * dx)
# L = dlx.fem.form(ufl.inner(v, w) * dx)

# bcs = []
# A = dlx.fem.petsc.assemble_matrix(a, bcs)
# A.assemble()
# b = dlx.fem.petsc.assemble_vector(L)
# dlx.fem.petsc.apply_lifting(b, [a], [bcs])
# b.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)

# solver = petsc4py.PETSc.KSP().create(A.getComm())
# solver.setOperators(A)
# solver.solve(b, d.vector)


#write to file using Method-1 or Method-2:
with dlx.io.XDMFFile(msh.comm, "parallel_test_1_project_np{0:d}_X.xdmf".format(nproc),"w") as file:
    file.write_mesh(msh)
    file.write_function(d)

##################################################################
######## code from here onwards irrelevant to projection#########



#writing misfit


# with dlx.io.XDMFFile(msh.comm, "parallel_test_2_utrue_np{0:d}_X.xdmf".format(nproc),"w") as file:
#     file.write_mesh(msh)
#     file.write_function(u_true_func)

x_true_2 = [u_true,m_true,u_true]

u = vector2Function(x_true_2[STATE], Vh[STATE])
m = vector2Function(x_true_2[PARAMETER], Vh[PARAMETER])
p = vector2Function(x_true_2[ADJOINT], Vh[ADJOINT])
dm = ufl.TestFunction(Vh[PARAMETER])
res_form = pde_handler(u, m, p)

test_val = pde.generate_state()
# test_val = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, ufl.conj(dm))))

test_val = dlx.fem.petsc.assemble_vector( dlx.fem.form(ufl.derivative(res_form, m, dm)) )
# print(type(test_val))



######################Works########################
# adj = dlx.la.create_petsc_vector(Vh[ADJOINT].dofmap.index_map,Vh[ADJOINT].dofmap.index_map_bs)
# adj_rhs = dlx.fem.Function(Vh[ADJOINT])
# adj_rhs.interpolate(lambda x: np.log(4.3) + 4.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 7.) )  )
# adj_rhs.x.scatter_forward()
# adj_rhs = adj_rhs.vector

# pde.solveAdj(adj,x_true,adj_rhs)
# print(type(adj))
######################Works########################

# adj_func = vector2Function(adj,Vh[ADJOINT])
# #comparing serial and parallel results of solveAdj
# with dlx.io.XDMFFile(msh.comm, "serial_test_2_adjoint_np{0:d}.xdmf".format(nproc),"w") as file:
#     file.write_mesh(msh)
#     file.write_function(adj_func)

###############################################

# test_val_2 = dlx.fem.assemble_vector( dlx.fem.form(ufl.derivative(res_form, m, dm)))
# print(type(test_val))



test_val.ghostUpdate( addv= petsc4py.PETSc.InsertMode.ADD, mode = petsc4py.PETSc.ScatterMode.REVERSE)
test_val.ghostUpdate( addv= petsc4py.PETSc.InsertMode.INSERT,mode = petsc4py.PETSc.ScatterMode.FORWARD)

# print(len(np.array(test_val)))
# test_val.ghostUpdate()

# test_val

test_val_func = vector2Function(test_val,Vh[STATE])
# test_val_func.x.scatter_reverse()

# with dlx.io.XDMFFile(msh.comm, "parallel_test_5_evalgrad_np{0:d}.xdmf".format(nproc),"w") as file:
#     file.write_mesh(msh)
#     file.write_function(test_val_func)



# dlx.fem.assemble( ufl.derivative(res_form, m, dm), tensor=out)
# test_out = petsc4py.PETSc.Vec(dlx.fem.assemble_vector( dlx.fem.form(ufl.derivative(res_form, m, dm)))) #<class 'dolfinx.la.Vector'>
# test_out_func = vector2Function(test_out,Vh[STATE])

# dlx.la.create_petsc_vector(self.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 

# test_out = petsc4py.PETSc.Vec(dlx.fem.assemble_vector( dlx.fem.form(ufl.derivative(res_form, m, dm)))) #<class 'dolfinx.la.Vector'>

# test_out = dlx.fem.assemble_scalar(dlx.fem.form(ufl.derivative(res_form, m, dm))) #<class 'dolfinx.la.Vector'>



# print(type(test_out))

# val = dlx.fem.form(ufl.derivative(res_form, m, dm))

# test_val = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative(res_form, m, dm)))




# same plots in XDMF File when run using serial, parallel - just had scatter_forward in vec2func Function
# 1. python3 Diffusion_Approximation_example_4.py 
# 2. mpirun -n 4 python3 Diffusion_Approximation_example_4.py 

# with dlx.io.XDMFFile(msh.comm, "parallel_test_5_state_np{0:d}.xdmf".format(nproc),"w") as file:
#     file.write_mesh(msh)
#     file.write_function(u_true_func)


######################Works########################
# adj = dlx.la.create_petsc_vector(Vh[ADJOINT].dofmap.index_map,Vh[ADJOINT].dofmap.index_map_bs)
# adj_rhs = dlx.fem.Function(Vh[ADJOINT])
# adj_rhs.interpolate(lambda x: np.log(4.3) + 4.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 7.) )  )
# adj_rhs.x.scatter_forward()
# adj_rhs = adj_rhs.vector

# pde.solveAdj(adj,x_true,adj_rhs)
# print(type(adj))
######################Works########################

# adj_func = vector2Function(adj,Vh[ADJOINT])
# #comparing serial and parallel results of solveAdj
# with dlx.io.XDMFFile(msh.comm, "serial_test_2_adjoint_np{0:d}.xdmf".format(nproc),"w") as file:
#     file.write_mesh(msh)
#     file.write_function(adj_func)

###############################################

#PRIOR

# class VaritionalRegularization:
#     def __init__(self, Vh, functional_handler, isQuadratic=False):
#         self.Vh = Vh #Function space of the parameter.
#         self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
#         self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m
    
#     def cost(self, m):
#         1. Cast the petsc4py vector m to a dlx.Function mfun
#         2. Compute the cost by calling assemble on self.functional_handler(mfun)
#         3. Return the value of the cost (Make sure to call a AllReduce for parallel computations

#     def grad(self, m, out):
#         1. Cast the petsc4py vector m to a dlx.Function mfun
#         2. call symbolic differentation of self.functional_handler(mfun) wrt mfum
#         3. call assemble, update ghosts, and store the result in out

#     def setLinearizationPoint(self, m):
#       1. Cast the petsc4py vector m to a dlx.Function mfun
#       2. call symbolic differentiation (twice) to get the second variation of self.functional_handler(mfun) wrt mfun
#       3. assemble the Hessian operator (it's a sparse matrix!) in the attribute self.R
#       4. set up a linearsolver self.Rsolver that uses CG as Krylov method, gamg as preconditioner and self.R as operator


# #example of functional_handler:
# class H1TikhonvFunctional:
#     def __init__(self, gamma, delta, m0):
#         self.gamma = gamma #These are dlx Constant, Expression, or Function
#         self.delta = delta
#         self.m0 = m0

#     def __call__(self, m): #Here m is a dlx Function
#         return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
#                ufl.inner(self.delta * m, m)*ufl.dx


#before we implement regularization, should implement gradient evaluation function



