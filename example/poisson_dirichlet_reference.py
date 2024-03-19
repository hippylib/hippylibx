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
    def __init__(self, alpha : float, f : float):
        
        self.alpha = alpha
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

    Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
    Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]

    ndofs = [Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs, Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs ]
    master_print (comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print (comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs) )


    #dirichlet B.C.
    uD = dlx.fem.Function(Vh[hpx.STATE])
    uD.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
    uD.x.scatter_forward()
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dlx.mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = dlx.fem.locate_dofs_topological(Vh[hpx.STATE], fdim, boundary_facets)
    bc = dlx.fem.dirichletbc(uD, boundary_dofs)



    # FORWARD MODEL 
    alpha = 0.
    f = -6.
    pde_handler = Poisson_Approximation(alpha, f)  
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [bc], [],  is_fwd_linear=True)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)     
    m_true.interpolate(lambda x: 0. * x[0] )
    m_true.x.scatter_forward() 
    m_true = m_true.x

    u_true = pde.generate_state()  
    
    x_true = [u_true, m_true, None] 

    pde.solveFwd(u_true,x_true)
    
    uex = dlx.fem.Function(Vh[hpx.STATE])
    uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    uex.x.scatter_forward()
    # uex = uex.x

    xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    hpx.updateFromVector(xfun[hpx.STATE],u_true)
    u_fun_true = xfun[hpx.STATE]


    L2_error = dlx.fem.form(ufl.inner(u_fun_true - uex, u_fun_true - uex) * ufl.dx)
    error_local = dlx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))

    error_max = np.max(np.abs(uD.x.array-u_fun_true.x.array))
    # Only print the error on one process
    if comm.rank == 0:
        print(f"Error_L2 : {error_L2:.2e}") #Error_L2: 8.21e-16
        print(f"Error_max : {error_max:.2e}") #Error_max : 1.78e-15

    # # LIKELIHOOD``
    hpx.updateFromVector(xfun[hpx.STATE],u_true)
    u_fun_true = xfun[hpx.STATE]

    hpx.updateFromVector(xfun[hpx.PARAMETER],m_true)
    m_fun_true = xfun[hpx.PARAMETER]
    
    d = dlx.fem.Function(Vh[hpx.STATE])
    d.x.array[:] = u_true.array[:]
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance),d.x)
    d.x.scatter_forward()

    misfit_form = PoissonMisfitForm(d,noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form,[])

    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = 0.01
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(Vh_m,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)
    model = hpx.Model(pde, prior, misfit)

    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)    
    hpx.parRandom.normal(1.,noise)
    prior.sample(noise,m0)


    data_misfit_True = hpx.modelVerify(model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0))

    # data_misfit_False = hpx.modelVerify(model,m0,is_quadratic=False,misfit_only=False,verbose=(rank == 0))
   
    # # # #######################################
    
    # prior_mean_copy = prior.generate_parameter(0)
    # prior_mean_copy.array[:] = prior_mean.array[:]

    # x = [model.generate_vector(hpx.STATE), prior_mean_copy, model.generate_vector(hpx.ADJOINT)]

    # if rank == 0:
    #     print( sep, "Find the MAP point", sep)    
           
    # parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
    # parameters["rel_tolerance"] = 1e-6
    # parameters["abs_tolerance"] = 1e-9
    # parameters["max_iter"]      = 500
    # parameters["cg_coarse_tolerance"] = 5e-1
    # parameters["globalization"] = "LS"
    # parameters["GN_iter"] = 20
    # if rank != 0:
    #     parameters["print_level"] = -1
    
    # solver = hpx.ReducedSpaceNewtonCG(model, parameters)
    
    # x = solver.solve(x) 
    
    # if solver.converged:
    #     master_print(comm, "\nConverged in ", solver.it, " iterations.")
    # else:
    #     master_print(comm, "\nNot Converged")

    # master_print (comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    # master_print (comm, "Final gradient norm: ", solver.final_grad_norm)
    # master_print (comm, "Final cost: ", solver.final_cost)
    
    # optimizer_results = {}
    # if(solver.termination_reasons[solver.reason] == 'Norm of the gradient less than tolerance'):
    #     optimizer_results['optimizer']  = True
    # else:
    #     optimizer_results['optimizer'] = False


    # final_results = {"data_misfit_True":data_misfit_True,
    #                  "data_misfit_False":data_misfit_False,
    #                  "optimizer_results":optimizer_results}


    # return final_results


    #######################################

if __name__ == "__main__":    
    nx = 8
    ny = 8
    noise_variance = 1e-4
    prior_param = {"gamma": 0.1, "delta": 1.}
    run_inversion(nx, ny, noise_variance, prior_param)
    
    comm = MPI.COMM_WORLD
    if(comm.rank == 0):
        plt.savefig("poisson_result_FD_Gradient_Hessian_Check")
        plt.show()

