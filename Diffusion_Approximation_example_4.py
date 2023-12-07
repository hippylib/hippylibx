# Attempting to fix solveFwd that returns different plot when run in serial (python3 _.py) and parallel (mpirun -n 4 python3 _.py)

import dolfinx as dlx
from mpi4py import MPI
# import matplotlib.pyplot as plt
# import pyvista
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
    # fun.interpolate(lambda x: np.full((x.shape[1],),0.))
    fun.vector.axpy(1., x)
    return fun

@unused_function
def Transpose(A):
    A.assemble()
    AT = petsc4py.PETSc.Mat()
    AT = A.transpose()
    rmap,cmap = A.getLGMap()
    AT.setLGMap(cmap,rmap)
    return AT 


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
        loc_cost = dlx.fem.assemble_scalar(dlx.fem.form(self.form(x[STATE], x[PARAMETER])))
        return self.mesh.comm.allreduce(loc_cost,op=MPI.SUM)



    def grad(self, i, x):
        x_fun = [x[STATE],x[PARAMETER]] 
        loc_grad = dlx.fem.assemble.assemble_vector(dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i])) ) #<class 'dolfinx.la.Vector'>
        loc_grad.scatter_forward()
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
        loc_action = dlx.fem.assemble.assemble_vector(dlx.fem.form(action) ) #<class 'dolfinx.la.Vector'>
        loc_action.scatter_forward()
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
        return dlx.fem.Function(self.Vh[STATE]).vector

    @unused_function
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dlx.fem.Function(self.Vh[PARAMETER]).vector()
    
    @unused_function   
    def init_parameter(self, m):
        """ Initialize the parameter."""
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.dofmap.index_map)

    def solveFwd(self, state, x): #state is a vector
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        
        result = vector2Function(state,Vh[STATE])

        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        if self.is_fwd_linear:    
            u = ufl.TrialFunction(self.Vh[STATE])
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, vector2Function(x[PARAMETER],Vh[PARAMETER]), p) #all 3 arguments-dl.Function types
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)

            ##fenicsX from here######
            
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form),bcs=self.bc)
            A.assemble() #petsc4py.PETSc.Mat

            self.solver.setOperators(A)

            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))

            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            dlx.fem.petsc.set_bc(b,self.bc)
            b.assemble() 

            self.solver.solve(b,result.vector)
            result.x.scatter_forward()

            return result #returning a dlx.fem.Function, the .vector attribute (petsc4py vector) contains the
            #the values of the solution vector
         
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
        varf = self.varf_handler(x[STATE], x[PARAMETER], p) 
        adj_form = ufl.derivative(ufl.derivative(varf, x[STATE], du), p, dp)

        Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs=self.bc0)
        Aadj.assemble()
        self.solver.setOperators(Aadj)
        self.solver.solve(adj_rhs,adj.vector)
   
    def _createLUSolver(self):   
        ksp = petsc4py.PETSc.KSP().create()
        pc = ksp.getPC()
        ksp.setFromOptions()
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

msh = dlx.cpp.mesh.Mesh
fname = 'meshes/circle.xdmf'
fid = dlx.io.XDMFFile(comm,fname,"r")
msh = fid.read_mesh(name='mesh')

Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
Vh = [Vh_phi, Vh_m, Vh_phi]

u0 = 1.
D = 1./24.

# GROUND TRUTH
m_true = dlx.fem.Function(Vh_m)


m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>

#Works - in serial and parallel.
# with dlx.io.XDMFFile(msh.comm, "testing_m_vector_dolfinx_parallel_try7.xdmf","w") as file:
#     file.write_mesh(msh)
#     file.write_function(m_true)

m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>

pde_handler = DiffusionApproximation(D, u0, ufl.ds) #returns a ufl form
pde = PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)
u_true = pde.generate_state() #a vector, not a function, <class 'petsc4py.PETSc.Vec'>


x_true = [u_true, m_true, None]

u_true_func = pde.solveFwd(u_true, x_true) #dlx.fem.Function

#Both values same as expected from Fenics:
# print(u_true_func.vector[:].min())
# print(u_true_func.vector[:].max())



#different plots in XDMF File when run using:
#1. python3 Diffusion_Approximation_example_4.py 
#2. mpirun -n 4 python3 Diffusion_Approximation_example_4.py 
with dlx.io.XDMFFile(msh.comm, "parallel_forward_try_8.xdmf","w") as file:
    file.write_mesh(msh)
    file.write_function(u_true_func)

