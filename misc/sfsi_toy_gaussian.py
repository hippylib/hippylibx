import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os
import matplotlib
import dolfinx.fem.petsc

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
        
    def __call__(self,u,m):   

        return .5/self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx


class H1TikhonvFunctional:
    def __init__(self, gamma, delta):
        self.gamma = gamma #These are dlx Constant, Expression, or Function
        self.delta = delta

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
    m_true = m_true.x

    u_true = pde.generate_state()   #a vector, not a function, <class 'dolfinx.la.Vector'>
    
    x_true = [u_true, m_true, None]  #list of dlx.la.vectors    

    pde.solveFwd(u_true,x_true)

############################################################

    # LIKELIHOOD
    u_fun = hpx.vector2Function(u_true,Vh[hpx.STATE])
    m_fun = hpx.vector2Function(m_true,Vh[hpx.PARAMETER])
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)
    # print(d.x.array.min(),":",d.x.array.max())
    hpx.parRandom(comm).normal_perturb(np.sqrt(noise_variance),d.x)
    d.x.scatter_forward()
    
    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    #Method - original
    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = 0.01
    prior_mean = prior_mean.x

    # prior = hpx.BiLaplacianPrior(Vh_m,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)

    # noise = prior.generate_parameter("noise")
    # m0 = prior.generate_parameter(0)    
    # hpx.parRandom(comm).normal(1.,noise)
    # prior.sample(noise,m0)
    m0 = m_true

    #the methods in old prior have to be in new prior
    
    prior_functional = H1TikhonvFunctional(3.,4)
    prior = hpx.VariationalRegularization(msh, Vh_m, prior_functional)
   
    model = hpx.Model(pde, prior, misfit)

    eps, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=False,verbose=(rank == 0),eps=None)
    if(rank == 0):
        print(err_grad)

##################################################################


if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)
