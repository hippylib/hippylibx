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

class DiffusionApproximation:
    def __init__(self, D : float, u0 : float, ds : ufl.measure.Measure):
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
        
    def __call__(self, u: dlx.fem.Function, m : dlx.fem.Function, p : dlx.fem.Function) -> ufl.form.Form:
        return ufl.inner(self.D*ufl.grad(u), ufl.grad(p))*ufl.dx + \
            ufl.exp(m)*ufl.inner(u,p)*ufl.dx + \
            .5*ufl.inner(u-self.u0,p)*self.ds

class PACTMisfitForm:
    def __init__(self, d : float, sigma2 : float):
        self.sigma2 = sigma2
        self.d = d
        
    def __call__(self,u : dlx.fem.Function, m : dlx.fem.Function) -> ufl.form.Form:   
        return .5/self.sigma2*ufl.inner(u*ufl.exp(m) -self.d, u*ufl.exp(m) -self.d)*ufl.dx


class H1TikhonvFunctional:
    def __init__(self, gamma, delta):
        self.gamma = gamma #These are dlx Constant, Expression, or Function
        self.delta = delta

    def __call__(self, m : dlx.fem.Function) -> ufl.form.Form: #Here m is a dlx Function
        return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
        ufl.inner(self.delta * m, m)*ufl.dx
        

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

    # ndofs = [Vh_phi.dim(), Vh_m.dim()]
    ndofs = [Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs, Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs ]
    master_print (comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print (comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs) )

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

    xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    # LIKELIHOOD
    hpx.updateFromVector(xfun[hpx.STATE],u_true)
    u_fun_true = xfun[hpx.STATE]

    hpx.updateFromVector(xfun[hpx.PARAMETER],m_true)
    m_fun_true = xfun[hpx.PARAMETER]
    
    #does it look correct??
    # with dlx.io.XDMFFile(msh.comm, "TEST_m0_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(m_fun) 

    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun_true * ufl.exp(m_fun_true)
    hpx.projection(expr,d)
    hpx.parRandom(comm).normal_perturb(np.sqrt(noise_variance),d.x)
    d.x.scatter_forward()

    misfit_form = PACTMisfitForm(d, noise_variance)
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

    # print(m0.array.min(),":",m0.array.max())

    # m0 = dlx.fem.Function(Vh_m) 
    # m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m0.x.scatter_forward() 
    # m0 = m0.x


    #######################################
    hpx.modelVerify(comm,model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0),eps=None)
    if(rank == 0):
        plt.show()    

    prior_mean_copy = prior.generate_parameter(0)
    prior_mean_petsc_vec = dlx.la.create_petsc_vector_wrap(prior_mean)

    temp_petsc_object = dlx.la.create_petsc_vector_wrap(prior_mean_copy)
    temp_petsc_object.scale(0.)  
    temp_petsc_object.axpy(1.,prior_mean_petsc_vec)

    temp_petsc_object.destroy()
    prior_mean_petsc_vec.destroy()

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
    #failing with multiple procs: fix this
    x = solver.solve(x)
    
    # if solver.converged:
    #     master_print(comm, "\nConverged in ", solver.it, " iterations.")
    # else:
    #     master_print(comm, "\nNot Converged")

    # master_print (comm, "Termination reason: ", solver.termination_reasons[solver.reason])
    # master_print (comm, "Final gradient norm: ", solver.final_grad_norm)
    # master_print (comm, "Final cost: ", solver.final_cost)
    # #######################################

    # # x = [u_true,m0,None]

    # # m_fun        = hpx.vector2Function(x[hpx.PARAMETER], Vh_m, name = "m_map")
    # # u_fun        = hpx.vector2Function(x[hpx.STATE], Vh_phi, name = "u_map")
    # # m_true_fun   = hpx.vector2Function(m_true, Vh_m, name = "m_true")
    # # u_true_fun   = hpx.vector2Function(u_true, Vh_phi, name = "u_true")

    # hpx.updateFromVector(xfun[hpx.PARAMETER],x[hpx.PARAMETER])    
    # m_fun = xfun[hpx.PARAMETER].copy()
    # m_fun.name = 'm_map'

    # hpx.updateFromVector(xfun[hpx.STATE],x[hpx.STATE])    
    # u_fun = xfun[hpx.STATE].copy()
    # u_fun.name = 'u_map'

    # hpx.updateFromVector(xfun[hpx.PARAMETER],m_true)    
    # m_true_fun = xfun[hpx.PARAMETER].copy()
    # m_true_fun.name = 'm_true'

    # hpx.updateFromVector(xfun[hpx.STATE],u_true)    
    # u_true_fun = xfun[hpx.STATE].copy()
    # u_true_fun.name = 'u_true'

    # d.name = 'data'

    # # obs_fun = dl.project(u_fun*m_fun, Vh[hp.STATE])

    # obs_fun = dlx.fem.Function(Vh[hpx.STATE],name='obs')
    # expr = u_fun * m_fun
    # hpx.projection(expr,obs_fun)
    # obs_fun.x.scatter_forward()


    # fid = dlx.io.XDMFFile(msh.comm,"m_map_6.xdmf","w")
    # fid.write_mesh(msh)
    # fid.write_function(m_fun,0)
    # fid.write_function(m_true_fun,0)
    # fid.write_function(u_fun,0)
    # fid.write_function(u_true_fun,0)
    # fid.write_function(d,0)
    # fid.write_function(obs_fun,0)

    # model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
    # Hmisfit = hpx.ReducedHessian(model, misfit_only=True)
    # k = 80
    # p = 20

    # if rank == 0:
    #     print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
    
    # Omega = hpx.MultiVector(x[hp.PARAMETER], k+p)
    # hp.parRandom.normal(1., Omega)

    # d, U = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    
    # U.export(Vh[hp.PARAMETER], "results/evect.xdmf", varname = "gen_evects", normalize = True)
    # if rank == 0:
    #     np.savetxt("results/eigevalues.dat", d)

    # if rank == 0:
    #     plt.figure()
    #     plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
    #     plt.yscale('log')
    
    
    # if rank == 0: 
    #     plt.show()



    # with dlx.io.XDMFFile(msh.comm, "TEST_m0_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(m_fun) 


    # test_func = dlx.fem.Function(Vh[hpx.STATE],name='dog')

    # test_func = dlx.fem.Function(Vh[hpx.STATE])
    # hpx.updateFromVector_2(test_func, u_true)
    
    # updateFromVector(self.xfun[PARAMETER], x[PARAMETER])
    # mfun = self.xfun[PARAMETER]

if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)

