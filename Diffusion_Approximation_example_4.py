# Attempting to fix solveFwd that returns different plot when run in serial (python3 _.py) and parallel (mpirun -n 4 python3 _.py)

import dolfinx as dlx
import dolfinx.fem.petsc
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
    print("Casting vector x to a function\n")
    fun.vector.axpy(1., x)
    # Make sure fun is ready to be used for assembly or plotting
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
        #NO!!! x is a list of PETSc.Vec so you first need to call vector2function on them
        loc_cost = dlx.fem.assemble_scalar(dlx.fem.form(self.form(x[STATE], x[PARAMETER])))
        return self.mesh.comm.allreduce(loc_cost,op=MPI.SUM)



    def grad(self, i, x):
	#NO!!! x is a list of PETSc.Vec so you first need to call vector2function on them
        x_fun = [x[STATE],x[PARAMETER]] 
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

	
        mfun = vector2Function(x[PARAMETER],Vh[PARAMETER])
        print("mfun created")

        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        print("Solver created")

        if self.is_fwd_linear:    
            u = ufl.TrialFunction(self.Vh[STATE])
            p = ufl.TestFunction(self.Vh[ADJOINT])

            res_form = self.varf_handler(u, mfun, p) #all 3 arguments-dl.Function types
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)

            ##fenicsX from here######
            
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form),bcs=self.bc)
            A.assemble() #petsc4py.PETSc.Mat
            print("A created")

            self.solver.setOperators(A)
            print("Operator set")

            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))

            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            dlx.fem.petsc.set_bc(b,self.bc)
            print("RHS assembled")

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
        #NO! x[STATE], x[PARAMETER] need to be converted to Function first
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
m_true.x.scatter_forward()

#Works - in serial and parallel.
with dlx.io.XDMFFile(msh.comm, "parameter_np{0:d}.xdmf".format(nproc),"w") as file:
     file.write_mesh(msh)
     file.write_function(m_true)

m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>

pde_handler = DiffusionApproximation(D, u0, ufl.ds) #returns a ufl form
pde = PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)
u_true = pde.generate_state() #a vector, not a function, <class 'petsc4py.PETSc.Vec'>


x_true = [u_true, m_true, None]

print("Enter solveFwd")
pde.solveFwd(u_true, x_true) #dlx.fem.Function
print("Completed solveFwd")

u_true_func = vector2Function(u_true, Vh[STATE]) 
#different plots in XDMF File when run using:
#1. python3 Diffusion_Approximation_example_4.py 
#2. mpirun -n 4 python3 Diffusion_Approximation_example_4.py 
with dlx.io.XDMFFile(msh.comm, "state_np{0:d}.xdmf".format(nproc),"w") as file:
    file.write_mesh(msh)
    file.write_function(u_true_func)

