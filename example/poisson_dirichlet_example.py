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
    Vh_phi = dlx.fem.FunctionSpace(msh, ("Lagrange", 2)) 
    Vh_m = dlx.fem.FunctionSpace(msh, ("Lagrange", 1))
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

    with dlx.io.XDMFFile(msh.comm, "dirichlet_poisson_BiLaplacian_Prior_true_parameter_np{0:d}_X.xdmf".format(nproc),"w") as file:
        file.write_mesh(msh)
        file.write_function(m_true) 

  
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

    data_misfit_True = hpx.modelVerify(model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0))

    data_misfit_False = hpx.modelVerify(model,m0,is_quadratic=False,misfit_only=False,verbose=(rank == 0))
   
    # # #######################################
    
    initial_guess_m = prior.generate_parameter(0)
    initial_guess_m.array[:] = prior_mean.array[:]
    
    x = [model.generate_vector(hpx.STATE), initial_guess_m, model.generate_vector(hpx.ADJOINT)]
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

    m_fun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER],name = 'm_map')
    m_true_fun = hpx.vector2Function(m_true, Vh[hpx.PARAMETER], name = 'm_true')
    
    V_P1 = dlx.fem.FunctionSpace(msh, ("Lagrange", 1) )

    u_true_fun = dlx.fem.Function(V_P1, name='u_true')
    u_true_fun.interpolate( hpx.vector2Function(u_true, Vh[hpx.STATE] ) )
    u_true_fun.x.scatter_forward()

    u_map_fun = dlx.fem.Function(V_P1, name='u_map')
    u_map_fun.interpolate( hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE] ) )
    u_map_fun.x.scatter_forward()
    
    d_fun = dlx.fem.Function(V_P1, name='data')
    d_fun.interpolate(d)
    d_fun.x.scatter_forward()

    with dlx.io.VTXWriter(msh.comm, "poisson_Dirichlet_BiLaplacian_prior_np{0:d}_Prior.bp".format(nproc), [m_fun, m_true_fun, u_map_fun, u_true_fun, d_fun] ) as vtx:
        vtx.write(0.0)

    if solver.converged:
        master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        master_print(comm, "\nNot Converged")
    master_print (comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    master_print (comm, "Final gradient norm: ", solver.final_grad_norm)
    master_print (comm, "Final cost: ", solver.final_cost)
    
    optimizer_results = {}
    if(solver.termination_reasons[solver.reason] == 'Norm of the gradient less than tolerance'):
        optimizer_results['optimizer']  = True
    else:
        optimizer_results['optimizer'] = False


    final_results = {"data_misfit_True":data_misfit_True,
                     "data_misfit_False":data_misfit_False,
                     "optimizer_results":optimizer_results}
    
    return final_results
    #######################################


if __name__ == "__main__":    
    nx = 64
    ny = 64
    noise_variance = 1e-4
    prior_param = {"gamma": 0.03, "delta": 0.3}
    run_inversion(nx, ny, noise_variance, prior_param)
    
    comm = MPI.COMM_WORLD
    if(comm.rank == 0):
        plt.savefig("poisson_result_FD_Gradient_Hessian_Check")
        plt.show()