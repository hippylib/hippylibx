#Trying to replicate example here: https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
#Poisson example with DirichletBC on left and right, 
# Neumann BC on top and Robin BC on bottom on the 2d square mesh with 
# Right -> u = ud 
# Left -> u = ud
# Top -> Neumann
# Bottom -> Robin
# using BiLaplacian Prior. 


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
    # Poisson_Approximation(g_Neumann, f,ds_boundaries,robin_r,robin_s)  
    def __init__(self, alpha : float, f : float, ds_boundaries, robin_r, robin_s):
        
        self.alpha = alpha
        self.f = f
        # self.dx = ufl.Measure("dx",metadata={"quadrature_degree":4})
        # self.ds = ufl.Measure("ds",metadata={"quadrature_degree":4})
        self.dx = ufl.Measure("dx")
        self.ds = ufl.Measure("ds")
           
        self.ds_boundaries = ds_boundaries     
        self.robin_r = robin_r
        self.robin_s = robin_s
        
    def __call__(self, u: dlx.fem.Function, m : dlx.fem.Function, p : dlx.fem.Function) -> ufl.form.Form:
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p))*self.dx - ufl.inner(self.f,p)*self.dx \
            + ufl.inner(self.alpha,p)*self.ds_boundaries(4) \
                + self.robin_r* ufl.inner(u - self.robin_s, p)*self.ds_boundaries(3) 


class PoissonMisfitForm:
    def __init__(self, d : float, sigma2 : float):
        self.d = d
        self.sigma2 = sigma2
        # self.dx = ufl.Measure("dx",metadata={"quadrature_degree":4})
        self.dx = ufl.Measure("dx")

    def __call__(self, u : dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:   
        return .5/self.sigma2*ufl.inner(u - self.d, u - self.d)*self.dx

def run_inversion(nx : int, ny : int, noise_variance : float, prior_param : dict) -> None:
    sep = "\n"+"#"*80+"\n"    

    comm = MPI.COMM_WORLD
    rank  = comm.rank
    nproc = comm.size

    # msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
    msh = dlx.mesh.create_unit_square(comm, nx, ny)
        
    Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
    Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]

    ndofs = [Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs, Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs ]
    master_print (comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print (comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs) )

    u_ex = lambda x: 1 + x[0]**2 + 2 *x[1]**2
    x_spatial = ufl.SpatialCoordinate(msh)
    robin_s = u_ex(x_spatial)
    f = -ufl.div(ufl.grad(u_ex(x_spatial)))
    normal = ufl.FacetNormal(msh)
    g_Neumann = -ufl.dot(normal, ufl.grad(u_ex(x_spatial)))
    robin_r = dlx.fem.Constant(msh, dlx.default_scalar_type(1000))

    #boundaries
    boundaries = [(1, lambda x: np.isclose(x[0], 0)), 
                  (2, lambda x: np.isclose(x[0], 1)),  
                  (3, lambda x: np.isclose(x[1], 0)),  
                  (4, lambda x: np.isclose(x[1], 1))] 

    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = dlx.mesh.locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dlx.mesh.meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    # with dlx.io.XDMFFile(msh.comm, "facet_tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_meshtags(facet_tag, msh.geometry)

    # ds_boundaries = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag, metadata={"quadrature_degree":4})
    ds_boundaries = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    class BoundaryCondition():
        def __init__(self, type, marker, values):
            self._type = type
            if type == "Dirichlet":
                u_D = dlx.fem.Function(Vh[hpx.STATE])
                u_D.interpolate(values)
                facets = facet_tag.find(marker)
                dofs = dlx.fem.locate_dofs_topological(Vh[hpx.STATE], fdim, facets)
                self._bc = dlx.fem.dirichletbc(u_D, dofs)
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type

    #dirichletBC:
    # u_left = dlx.fem.Function(Vh[hpx.STATE])
    # u_left.interpolate(lambda x: np.full((x.shape[1],),0.))
    # u_left.x.scatter_forward()

    # u_right = dlx.fem.Function(Vh[hpx.STATE])
    # u_right.interpolate(lambda x: np.full((x.shape[1],),1.))
    # u_right.x.scatter_forward()

    # Define the Dirichlet condition
    boundary_conditions = [BoundaryCondition("Dirichlet", 1, u_ex),
                        BoundaryCondition("Dirichlet", 2, u_ex)]

    bc = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bc.append(condition.bc)

    pde_handler = Poisson_Approximation(g_Neumann, f,ds_boundaries,robin_r,robin_s)  

    #bc0
    tdim_bc0 = msh.topology.dim
    fdim_bc0 = tdim_bc0 - 1
    msh.topology.create_connectivity(fdim_bc0, tdim_bc0)
    boundary_facets_bc0 = dlx.mesh.exterior_facet_indices(msh.topology)
    boundary_dofs_bc0 = dlx.fem.locate_dofs_topological(Vh[hpx.STATE], fdim, boundary_facets_bc0)
    uD_0 = dlx.fem.Function(Vh[hpx.STATE])
    uD_0.interpolate(lambda x: 0. * x[0])
    uD_0.x.scatter_forward()
    bc0 = dlx.fem.dirichletbc(uD_0,boundary_dofs_bc0)

    # # FORWARD MODEL 
    # f = -6.

    pde = hpx.PDEVariationalProblem(Vh, pde_handler, bc, [bc0],  is_fwd_linear=True)
    pde.petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)     
    m_true.interpolate(lambda x: 0. * x[0] )
    m_true.x.scatter_forward() 
    m_true = m_true.x

    u_true = pde.generate_state()  
    
    x_true = [u_true, m_true, None] 

    pde.solveFwd(u_true,x_true)
 
    u_fun_true = hpx.vector2Function(u_true,Vh[hpx.STATE])

    V_ex = dlx.fem.FunctionSpace(msh, ("Lagrange", 2))
    u_exact = dlx.fem.Function(V_ex)
    u_exact.interpolate(u_ex)

    error_L2 = np.sqrt(msh.comm.allreduce(dlx.fem.assemble_scalar(dlx.fem.form( (u_fun_true - u_exact)**2 * ufl.dx)), op=MPI.SUM))

    u_vertex_values = u_fun_true.x.array
    uex_1 = dlx.fem.Function(Vh_phi)
    uex_1.interpolate(u_ex)
    u_ex_vertex_values = uex_1.x.array
    error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
    error_max = msh.comm.allreduce(error_max, op=MPI.MAX)
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
 
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
    nx = 10
    ny = 10
    noise_variance = 1e-4
    prior_param = {"gamma": 0.1, "delta": 1.}
    run_inversion(nx, ny, noise_variance, prior_param)
    
    comm = MPI.COMM_WORLD
    if(comm.rank == 0):
        plt.savefig("poisson_result_FD_Gradient_Hessian_Check")
        plt.show()

