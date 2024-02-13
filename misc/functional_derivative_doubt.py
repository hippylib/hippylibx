import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os
import matplotlib
import dolfinx.fem.petsc


from matplotlib import pyplot as plt


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

    m_true = dlx.fem.Function(Vh_m) 
    m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m_true.x.scatter_forward() 
    m_true = m_true.x

    prior_functional = H1TikhonvFunctional(3.,4)

    mfun = hpx.vector2Function(m_true,Vh_m)

    # 2. call symbolic differentiation (twice) to get the second 
    # variation of self.functional_handler(mfun) wrt mfun

    L = ufl.derivative(ufl.derivative(prior_functional(mfun),mfun), mfun)
    L_form = dlx.fem.form(L)
    prior_R = dlx.fem.petsc.assemble_matrix(L_form)

if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)