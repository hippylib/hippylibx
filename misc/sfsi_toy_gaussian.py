import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os
import matplotlib

# matplotlib.use('Agg')

from matplotlib import pyplot as plt

# sys.path.append( os.environ.get('HIPPYLIBX_BASE_DIR', "../hippylibX") )

sys.path.append( os.environ.get('HIPPYLIBX_BASE_DIR', "../") )

import hippylibX as hpx


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

class PACTMisfitForm:
    def __init__(self, d, sigma2):
        self.sigma2 = sigma2
        self.d = d
        # self.mesh = mesh
        
    def __call__(self,u,m):   

        # return dlx.fem.Constant(self.mesh, petsc4py.PETSc.ScalarType(.5)) /self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx
        return .5/self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx
        # return dl.Constant(.5)/self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx


class H1TikhonvFunctional:
    def __init__(self, gamma, delta, m0):
        self.gamma = gamma #These are dlx Constant, Expression, or Function
        self.delta = delta
        self.m0 = m0

    def __call__(self, m): #Here m is a dlx Function
        return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
               ufl.inner(self.delta * m, m)*ufl.dx


def run_inversion(nx, ny, noise_variance, prior_param):
    sep = "\n"+"#"*80+"\n"    

    comm = MPI.COMM_WORLD
    rank  = comm.rank
    nproc = comm.size

    fname = '../example/meshes/circle.xdmf'
    fid = dlx.io.XDMFFile(comm,fname,"r")
    msh = fid.read_mesh(name='mesh')

    Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
    Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]

    # FORWARD MODEL
    u0 = 1.
    D = 1./24.
    pde_handler = DiffusionApproximation(D, u0, ufl.ds)     #returns a ufl form
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)
    m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m_true.x.scatter_forward() 


    #to preserve values after using solveFwd
    m_fun_true = dlx.fem.Function(Vh_m)
    m_fun_true.x.array[:] = m_true.x.array[:]

    m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>
    
    
    u_true = pde.generate_state()   #a vector, not a function, <class 'petsc4py.PETSc.Vec'>
    x_true = [u_true, m_true, None]     #list of petsc vectors    
    A_mat = pde.solveFwd(u_true, x_true) #petsc4py.PETSc.Vec

    
    # #to preserve values in x_true        
    x_true[hpx.STATE] = u_true
    x_true[hpx.PARAMETER] = m_fun_true.vector

    m_true = x_true[hpx.PARAMETER]

    adj_vec = pde.generate_state()
    #a test vector
    adj_rhs = pde.generate_state()

    
    adj_rhs = dlx.fem.Function(Vh_m)
    adj_rhs.interpolate(lambda x: np.log(0.34) + 3.*( ( ( (x[0]-1.5)*(x[0]-1.5) + (x[1]-1.5)*(x[1]-1.5) ) < 0.75) )) # <class 'dolfinx.fem.function.Function'>
    adj_rhs.x.scatter_forward() 

    ##############################   
    pde.solveAdj_2(adj_vec, x_true, adj_rhs)
    x_true[hpx.ADJOINT] = adj_vec

    adj_vec_func = hpx.vector2Function(adj_vec,Vh[hpx.ADJOINT])   

    grad_val = pde.evalGradientParameter(x_true)
    
    #same values in serial and parallel
    
    grad_func = hpx.vector2Function(grad_val,Vh[hpx.STATE])
    
    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    
    # # LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)

    d.x.scatter_forward()
    
    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    # # PRIOR
    grad = misfit.grad(0,x_true) #different plots in serial and parallel

    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])

    trial = ufl.TrialFunction(Vh_phi)
    test  = ufl.TestFunction(Vh_phi)
    varfM = ufl.inner(trial,test)*ufl.dx       
    M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM)) #used as set_Operator, so must be matrix

    qdegree = 2*Vh_phi._ufl_element.degree()
    metadata = {"quadrature_degree" : qdegree}

    num_sub_spaces = Vh_phi.num_sub_spaces #0

    element = ufl.FiniteElement("Quadrature", Vh_phi.mesh.ufl_cell(), qdegree, quad_scheme="default")
    
    Qh = dlx.fem.FunctionSpace(Vh_phi.mesh, element)

    ph = ufl.TrialFunction(Qh)
    qh = ufl.TestFunction(Qh)
    Mqh = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,qh)*ufl.dx(metadata=metadata)) ) #petsc4py.PETSc.Mat
    
    ones = Mqh.createVecRight()
    ones.set(1.)
    dMqh = Mqh.createVecLeft()
    Mqh.assemble()
    Mqh.mult(ones,dMqh)

    dMqh.setArray(ones.getArray()/np.sqrt(dMqh.getArray()))
        
    Mqh.setDiagonal(dMqh)
    
    MixedM = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,test)*ufl.dx(metadata=metadata)))
    MixedM.assemble()

    sqrtM = MixedM.matMult(Mqh)
    sqrtM.assemble() #may not be needed

    prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"])
    
    model = hpx.Model(pde, prior, misfit)

    noise = prior.init_vector("noise")

    m0 = prior.init_vector(0)

    noise.set(3.)

    prior.sample(noise,m0)

    # modify m0 so you can have non-zero vector solution of x[STATE]    
    m0 = dlx.fem.Function(Vh_m)
    m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m0.x.scatter_forward() 
    
    # m_fun_true = dlx.fem.Function(Vh_m)
    # m_fun_true.x.array[:] = m0.x.array[:]

    # m0 = m0.vector
    # is_quadratic = False

    _, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0),eps=None)

    # eps, err_grad, _ = hpx.modelVerify(comm, model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))

    # index = 2
    # h = model.generate_vector(hpx.PARAMETER)

    # h.set(5.)
   
    # x = model.generate_vector()


    # model.solveFwd(x[hpx.STATE], x)
    
    # x[hpx.PARAMETER] = m_fun_true.vector
    # m0 = m_fun_true.vector
   
    # rhs = model.problem.generate_state()
    
    # ##########################
    # u_fun = hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE])
    # m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER])
    # x_fun = [u_fun, m_fun]
    # x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]

    # L = dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[hpx.STATE], x_test[hpx.STATE]))

    # rhs = dlx.fem.petsc.create_vector(L)

    # with rhs.localForm() as loc_grad:
    #     loc_grad.set(0)

    # dlx.fem.petsc.assemble_vector(rhs,L)

    # rhs.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    # ##########################
    
    # rhs.scale(-1.) #slight difference in values.

    # # print(rank,":",rhs.getArray().min(),":",rhs.getArray().max())

    # rhs_func = hpx.vector2Function(rhs,Vh[hpx.ADJOINT])
    # model.problem.solveAdj_2(x[hpx.ADJOINT],x,rhs_func)
    
    # # print(x[hpx.ADJOINT].getArray().min(),":",x[hpx.ADJOINT].getArray().max())    

    # cx = model.cost(x)

    # grad_x = model.generate_vector(hpx.PARAMETER)
    # calc_val,grad_x = model.evalGradientParameter(x,misfit_only=True)

    # grad_xh = grad_x.dot( h )

    # eps = None

    # if eps is None:
    #     n_eps = 32
    #     eps = np.power(.5, np.arange(n_eps))
    #     eps = eps[::-1]
    # else:
    #     n_eps = eps.shape[0]
    
    # err_grad = np.zeros(n_eps)
    # err_H = np.zeros(n_eps)

    # for i in range(n_eps):
    #     my_eps = eps[i]
        
    #     x_plus = model.generate_vector()
    #     x_plus[hpx.PARAMETER].axpy(1., m0 )
    #     x_plus[hpx.PARAMETER].axpy(my_eps, h)
        
    #     model.solveFwd(x_plus[hpx.STATE], x_plus)

    #     dc = model.cost(x_plus)[index] - cx[index]
        
    #     err_grad[i] = abs(dc/my_eps - grad_xh)

    ####################################################
    
    ############################################################################
    
    
    if(rank == 0):
        print(err_grad)
    #####################################################

if __name__ == "__main__":    
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.05, "delta": 1.}
    
    run_inversion(nx, ny, noise_variance, prior_param)

 