#Poisson example with DirichletBC on the 2d square mesh with 
# u_d = 1 on top, 0 on bottom using BiLaplacian Prior. 

import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os
import dolfinx.fem.petsc

from matplotlib import pyplot as plt

sys.path.append( os.environ.get('HIPPYLIBX_BASE_DIR', "../") )
import hippylibX as hpx

def master_print(comm, *args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)

class Poisson_Approximation:
    def __init__(self, f : float):
        
        self.f = f
        self.dx = ufl.Measure("dx",metadata={"quadrature_degree":4})
        
    def __call__(self, u: dlx.fem.Function, m : dlx.fem.Function, p : dlx.fem.Function) -> ufl.form.Form:
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p))*self.dx - self.f*p*self.dx 
    
class PoissonMisfitForm:
    def __init__(self, d : float, sigma2 : float):
        self.d = d
        self.sigma2 = sigma2
        self.dx = ufl.Measure("dx",metadata={"quadrature_degree":4})

    def __call__(self, u : dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:   
        return .5/self.sigma2*ufl.inner(u - self.d, u - self.d)*self.dx

def run_inversion(nx : int, ny : int, noise_variance : float, prior_param : dict) -> None:
    sep = "\n"+"#"*80+"\n"    

    comm = MPI.COMM_WORLD
    rank  = comm.rank
    nproc = comm.size

    msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)    

    Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 2)) 
    Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]

    ndofs = [Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs, Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs ]
    master_print (comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print (comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs) )

    #dirichlet B.C.
    uD = dlx.fem.Function(Vh[hpx.STATE])
    uD.interpolate(lambda x: x[1])
    uD.x.scatter_forward()

    def top_bottom_boundary(x):
        return np.logical_or(np.isclose(x[1],1), np.isclose(x[1],0))

    fdim = msh.topology.dim - 1
    top_bottom_boundary_facets = dlx.mesh.locate_entities_boundary(msh, fdim, top_bottom_boundary)
    dirichlet_dofs = dlx.fem.locate_dofs_topological(Vh[hpx.STATE], fdim, top_bottom_boundary_facets)
    bc = dlx.fem.dirichletbc(uD, dirichlet_dofs)

    #bc0
    uD_0 = dlx.fem.Function(Vh[hpx.STATE])
    uD_0.interpolate(lambda x: 0. * x[0])
    uD_0.x.scatter_forward()
    bc0 = dlx.fem.dirichletbc(uD_0,dirichlet_dofs)

    # # FORWARD MODEL 
    f = dlx.fem.Constant(msh,dlx.default_scalar_type(0.0))
    pde_handler = Poisson_Approximation(f)  
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [bc], [bc0],  is_fwd_linear=True)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)     
    m_true.interpolate(lambda x: np.log(2 + 7*( (    (x[0] - 0.5)**2 + (x[1] - 0.5)**2)**0.5 > 0.2)) )
    m_true.x.scatter_forward() 
    m_true = m_true.x

    u_true = pde.generate_state()  
    x_true = [u_true, m_true, None] 
    pde.solveFwd(u_true,x_true)
 
    # # LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    d.x.array[:] = u_true.array[:]
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance),d.x)
    d.x.scatter_forward()

    misfit_form = PoissonMisfitForm(d,noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form,[bc0])

    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = 0.01
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(Vh_m,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)
    model = hpx.Model(pde, prior, misfit)

    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)    
    hpx.parRandom.normal(1.,noise)
    prior.sample(noise,m0)

    #gives error
    # prior.R_petsc.mult(dlx.la.create_petsc_vector_wrap(m_true),dlx.la.create_petsc_vector_wrap(m_true))
    
    #works
    # prior.R_instance.mult(dlx.la.create_petsc_vector_wrap(m_true),dlx.la.create_petsc_vector_wrap(m_true)) - works

    #third attempt
    # prior.Rmult(dlx.la.create_petsc_vector_wrap(m_true),dlx.la.create_petsc_vector_wrap(m_true))

    # prior.R_petsc.mult(dlx.la.create_petsc_vector_wrap(m_true),dlx.la.create_petsc_vector_wrap(m_true))
    # help1 = prior.A.createVecLeft()
    # help2 = prior.A.createVecRight()
    
    # prior.Rmult(dlx.la.create_petsc_vector_wrap(m_true),dlx.la.create_petsc_vector_wrap(m_true))

    #######################################

if __name__ == "__main__":    
    nx = 64
    ny = 64
    noise_variance = 1e-4
    prior_param = {"gamma": 0.1, "delta": 1.}
    run_inversion(nx, ny, noise_variance, prior_param)
    
    comm = MPI.COMM_WORLD
    if(comm.rank == 0):
        plt.savefig("poisson_result_FD_Gradient_Hessian_Check")
        plt.show()

