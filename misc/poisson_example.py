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

# from memory_profiler import memory_usage
# from memory_profiler import profile

def master_print(comm, *args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)

class Poisson_Approximation:
    def __init__(self, alpha : float, f : float, ds : ufl.measure.Measure):
        
        self.alpha = alpha
        self.f = f
        self.ds = ds
        
    def __call__(self, u: dlx.fem.Function, m : dlx.fem.Function, p : dlx.fem.Function) -> ufl.form.Form:
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx + \
        self.alpha * ufl.inner(u,p)*ufl.ds  - self.f*p*ufl.dx 
    

class PoissonMisfitForm:
    def __init__(self, d : float, sigma2 : float):
        self.d = d
        self.sigma2 = sigma2
        
    def __call__(self, u : dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:   
        return .5/self.sigma2*ufl.inner(u - self.d, u - self.d)*ufl.dx


class H1TikhonvFunctional:
    def __init__(self, gamma, delta):
        self.gamma = gamma #These are dlx Constant, Expression, or Function
        self.delta = delta

    def __call__(self, m : dlx.fem.Function) -> ufl.form.Form: #Here m is a dlx Function
        return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
        ufl.inner(self.delta * m, m)*ufl.dx

# @profile
def run_inversion(nx : int, ny : int, noise_variance : float, prior_param : dict) -> None:
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

    ndofs = [Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs, Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs ]
    master_print (comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print (comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs) )

    # FORWARD MODEL
    alpha = 100.
    f = 1.
    pde_handler = Poisson_Approximation(alpha, f, ufl.ds)     #returns a ufl form
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)     
    m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m_true.x.scatter_forward() 
    m_true = m_true.x

    u_true = pde.generate_state()   #a vector, not a function, <class 'dolfinx.la.Vector'>
    
    x_true = [u_true, m_true, None]  #list of dlx.la.vectors    

    pde.solveFwd(u_true,x_true)

    xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    # LIKELIHOOD
    hpx.updateFromVector(xfun[hpx.STATE],u_true)
    u_fun_true = xfun[hpx.STATE]

    hpx.updateFromVector(xfun[hpx.PARAMETER],m_true)
    m_fun_true = xfun[hpx.PARAMETER]
    
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun_true
    hpx.projection(expr,d)
    hpx.parRandom(comm).normal_perturb(np.sqrt(noise_variance),d.x)
    d.x.scatter_forward()

    misfit_form = PoissonMisfitForm(d)
    misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = 0.01
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(Vh_m,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)
    model = hpx.Model(pde, prior, misfit)

    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)    
    hpx.parRandom(comm).normal(1.,noise)
    prior.sample(noise,m0)

    eps, err_grad, err_H = hpx.modelVerify(comm,model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0))
    
    if(rank == 0):
        print(err_grad,'\n')    
        print(err_H)
        plt.show()  

    #######################################
    
    prior_mean_copy = prior.generate_parameter(0)
    prior_mean_petsc_vec = dlx.la.create_petsc_vector_wrap(prior_mean)

    temp_petsc_object = dlx.la.create_petsc_vector_wrap(prior_mean_copy)
    temp_petsc_object.scale(0.)  
    temp_petsc_object.axpy(1.,prior_mean_petsc_vec)

    temp_petsc_object.destroy() #petsc vec of prior_mean_copy
    prior_mean_petsc_vec.destroy() #petsc vec of prior_mean

    x = [model.generate_vector(hpx.STATE), prior_mean_copy, model.generate_vector(hpx.ADJOINT)]

    if rank == 0:
        print( sep, "Find the MAP point", sep)    
           
    parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-6
    parameters["abs_tolerance"] = 1e-9
    parameters["max_iter"]      = 500
    parameters["cg_coarse_tolerance"] = 5e-1
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 20
    if rank != 0:
        parameters["print_level"] = -1
    
    solver = hpx.ReducedSpaceNewtonCG(model, parameters)
    
    x = solver.solve(x) 
    
    if solver.converged:
        master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        master_print(comm, "\nNot Converged")

    master_print (comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    master_print (comm, "Final gradient norm: ", solver.final_grad_norm)
    master_print (comm, "Final cost: ", solver.final_cost)

    #######################################


if __name__ == "__main__":    
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.05, "delta": 1.}
    run_inversion(nx, ny, noise_variance, prior_param)


