#MWE of error in H1TikhonovFunctional

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
    m_true = m_true.x

    u_true = pde.generate_state()   #a vector, not a function, <class 'dolfinx.la.Vector'>
    
    x_true = [u_true, m_true, None]  #list of dlx.la.vectors    

    pde.solveFwd(u_true,x_true)

    # print(rank,":",u_true.array.min(),":",u_true.array.max())

    # LIKELIHOOD
    u_fun = hpx.vector2Function(x_true[hpx.STATE],Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER],Vh[hpx.PARAMETER])
    
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)
    d.x.scatter_forward()
    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)


    prior_mean = dlx.fem.Function(Vh_m)
    # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.))
    prior_mean.x.array[:] = 0.01
    # prior_mean.x.scatter_forward() #not needed
    prior_mean = prior_mean.x


    prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)
  
    model = hpx.Model(pde, prior, misfit)

    noise = prior.init_vector("noise")
    m0 = prior.init_vector(0)    
    noise.array[:] = 3.
    prior.sample(noise,m0)


    #dummy example for non-zero values in x[STATE] after solveFwd
    m0 = dlx.fem.Function(Vh_m) 
    m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m0.x.scatter_forward() 
    m0 = m0.x

    index = 2
    h = model.generate_vector(hpx.PARAMETER)
    h.array[:] = 5
    x = model.generate_vector()

    x[hpx.PARAMETER] = m0 #dlx.la.Vector
    model.solveFwd(x[hpx.STATE], x)
    # model.solveAdj(x[hpx.ADJOINT], x ,Vh[hpx.ADJOINT])
    model.solveAdj(x[hpx.ADJOINT], x)


    #reg is supposed to be like misfit
    ufun = hpx.vector2Function(x[hpx.STATE],Vh[hpx.STATE])
    # misfit_form_reg = H1TikhonvFunctional(ufun,d,noise_variance)
    misfit_form_reg = H1TikhonvFunctional(3.,4.,noise_variance)
    
    mfun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER])
    loc_cost = misfit_form_reg(mfun)
    
    # glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))

    print(loc_cost,'\n')
        
    #compare loc_cost to one from misfit.py
    misfit_form_misfit = PACTMisfitForm(d, noise_variance)
    u_fun = hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER])
    loc_cost = misfit_form_misfit(u_fun,m_fun)
    
    print(loc_cost)

 

if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)
