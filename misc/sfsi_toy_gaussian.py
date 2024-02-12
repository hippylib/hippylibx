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
    def __init__(self, gamma, delta, m0):
        self.gamma = gamma #These are dlx Constant, Expression, or Function
        self.delta = delta
        self.m0 = m0

    def __call__(self, m): #Here m is a dlx Function
        return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
        ufl.inner(self.delta * m, m)*ufl.dx

        # return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + ufl.inner(self.delta * m, m)*ufl.dx
        # return self.delta * ufl.inner( self.gamma *ufl.exp(m), ufl.exp(m) ) *ufl.dx 
        # return ufl.inner(self.gamma * ufl.exp(m), self.gamma * ufl.exp(m) ) *ufl.dx 
    # + ufl.inner(self.delta * m, m)*ufl.dx

        #to make it similar to PACTMisfitForm - works
        # return .5/self.m0 * ufl.inner(self.gamma*ufl.exp(m) -self.delta, self.gamma*ufl.exp(m) -self.delta)*ufl.dx
        #make self.gamma = u, self.delta = d

        # return .5/self.m0 * ufl.inner(self.gamma*ufl.exp(m) -self.delta, self.gamma*ufl.exp(m) -self.delta)*ufl.dx
        


def run_inversion(nx, ny, noise_variance, prior_param):
    sep = "\n"+"#"*80+"\n"    

    comm = MPI.COMM_WORLD
    rank  = comm.rank
    nproc = comm.size

    fname = 'meshes/circle.xdmf'
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
    # print(rank,":",u_true.array.min(),":",u_true.array.max())

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

    # misfit_form = H1TikhonvFunctional(3.,4.,noise_variance)
    # misfit = hpx.VariationalRegularization(msh, Vh, misfit_form)
    # print(misfit.cost())

    #pde.solveAdj not working correctly

    # print(x_true[hpx.STATE].array.min(),":",x_true[hpx.STATE].array.max())
    # print(x_true[hpx.PARAMETER].array.min(),":",x_true[hpx.PARAMETER].array.max())


    # misfit_grad = misfit.grad(hpx.STATE,x_true)
    # print(misfit_grad.array.min(),":",misfit_grad.array.max())

    # misfit_cost = misfit.cost(x_true)
    
    # print(rank,":",misfit_cost)

    # p_true = pde.generate_state()
    # x_true[hpx.STATE] = u_true
    # x_true[hpx.PARAMETER] = m_true

    # # print(x_true[hpx.STATE].array.min(),":",x_true[hpx.STATE].array.max())
    # # print(x_true[hpx.PARAMETER].array.min(),":",x_true[hpx.PARAMETER].array.max())
    
    # rhs = misfit.grad(hpx.STATE, x_true)
        
    # dlx.la.create_petsc_vector_wrap(rhs).scale(-1.)
    # pde.solveAdj(p_true,x_true,dlx.la.create_petsc_vector_wrap(rhs))

    # print(p_true.array.min(),":",p_true.array.max())

    #plot
    # test_func = hpx.vector2Function(p_true,Vh[hpx.ADJOINT])
    # with dlx.io.XDMFFile(msh.comm, "ADJOINT_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(test_func)
    
    # PRIOR
    # prior_mean = dlx.fem.Function(Vh_m)
    # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.01))
    # prior_mean.x.scatter_forward()
    # prior_mean = prior_mean.x
    
    #Method - original
    prior_mean = dlx.fem.Function(Vh_m)
    # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.))
    prior_mean.x.array[:] = 0.01
    # prior_mean.x.scatter_forward() #not needed
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(Vh_m,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)
    # print(prior.cost(x_true[hpx.PARAMETER]))    


    #NEW PRIOR ATTEMPT
    # #################
    # new_prior = hpx.VariationalRegularization(msh,Vh_phi,H1TikhonvFunctional)
    # # p_handler = new_prior.functional_handler

    # m_true = dlx.fem.Function(Vh_m) 
    # m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m_true.x.scatter_forward() 
    # m_true = m_true.x

    # mfun = hpx.vector2Function(m_true,Vh[hpx.PARAMETER])
    # # print(m_true.array.min(),":",m_true.array.max())

    # # test_func_handler_obj = p_handler(3. ,4., 5.)

    # # ufl.derivative(test_func_handler_obj,ufl.TestFunction(Vh[hpx.PARAMETER]) 
    # # test_func_handler = H1TikhonvFunctional(3.,4.,5.)
    # # test_obj = test_func_handler(mfun)
    # # test_obj = ufl.derivative(test_func_handler(mfun),mfun)
    # # test_obj_2 = ufl.derivative(ufl.derivative(test_func_handler(mfun),mfun),mfun)
    # # test_obj_3 = dlx.fem.form(test_obj)
    # # test_obj = dlx.fem.form(test_func_handler(mfun))
    # # print(test_obj_2)
    # # print(type(test_obj))
    # #################


    #Fixing PRNG


    # prior_mean = dlx.fem.Function(Vh_m)
    # # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.))
    # prior_mean.x.array[:] = 0.01
    # # prior_mean.x.scatter_forward() #not needed
    # prior_mean = prior_mean.x


    # prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)

    # noise = prior.init_vector("noise")
    #create some dummy "noise" vector
    # noise = dlx.la.vector(Vh_phi.dofmap.index_map) #dummy value
    # noise = dlx.la.vector(Vh_phi.dofmap.index_map) #dummy value
    # print(len(noise.array))
    # prior.init_vector(noise, "noise")
    # print(len(noise.array))
    


    # x = dlx.la.vector(Vh_phi.dofmap.index_map)
    # print(len(x.array))
    # print(len(noise.array))


    # print(type(noise))
    # prior.init_vector(noise, "noise")
    # # print(len(noise.array))    
    # print(type(noise))
    # print(noise)

    #testing the methods in prior
    #init, sample, cost, grad

    model = hpx.Model(pde, prior, misfit)

    # noise = prior.init_vector("noise")
    # m0 = prior.init_vector(0)
    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)    
    # noise.array[:] = 3.
    hpx.parRandom(comm).normal(1.,noise)
    # print(noise.array.min(),":",noise.array.max())
    prior.sample(noise,m0)

    # print(m0.array.min(),":",m0.array.max())
    #create dummy m0:

    # m0 = dlx.fem.Function(Vh_m) 
    # m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m0.x.scatter_forward() 
    # m0 = m0.x

    eps, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=False,verbose=(rank == 0),eps=None)
    # if(rank == 0):
    #     print(err_grad)


##################################################################


    # #dummy example for non-zero values in x[STATE] after solveFwd
    # m0 = dlx.fem.Function(Vh_m) 
    # m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m0.x.scatter_forward() 
    # m0 = m0.x

    # # test_obj = dlx.fem.Function(Vh[hpx.PARAMETER])
    # # test_obj = test_obj.x

    # # pde.evalGradientParameter(x_true)
    
    # index = 2
    # h = model.generate_vector(hpx.PARAMETER)
    # h.array[:] = 5
    # x = model.generate_vector()

    # x[hpx.PARAMETER] = m0 #dlx.la.Vector
    # model.solveFwd(x[hpx.STATE], x)
    # # model.solveAdj(x[hpx.ADJOINT], x ,Vh[hpx.ADJOINT])
    # model.solveAdj(x[hpx.ADJOINT], x)

    # # test_pde_eval_grad = pde.evalGradientParameter(x)
    # test_pde_eval_grad = pde.generate_parameter()
    # pde.evalGradientParameter(x,test_pde_eval_grad)

    # # print(test_pde_eval_grad.array.min(),":",test_pde_eval_grad.array.max())
    # # print(len(test_pde_eval_grad.array))
    
    # test_func = hpx.vector2Function(test_pde_eval_grad,Vh[hpx.PARAMETER])
##################################################################

    #have plots to compare to when you modify:
    #True function plot - correct
    # with dlx.io.XDMFFile(msh.comm, "MODIFIED_PDE_EVAL_GRAD_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(test_func)

    #checking the values of below 3 in paraview:
    # print(rank,":",x[hpx.STATE].array.min(),":",x[hpx.STATE].array.max())
    # print(rank,":",x[hpx.PARAMETER].array.min(),":",x[hpx.PARAMETER].array.max())
    # print(rank,":",x[hpx.ADJOINT].array.min(),":",x[hpx.ADJOINT].array.max())
    #all correct plots in Paraview

    # print(rank,":",misfit.cost(x))
    # misfit_form = PACTMisfitForm(d, noise_variance)
    # misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    #reg is supposed to be like misfit
    # ufun = hpx.vector2Function(x[hpx.STATE],Vh[hpx.STATE])
    # # misfit_form_reg = H1TikhonvFunctional(ufun,d,noise_variance)
    # misfit_form_reg = H1TikhonvFunctional(3.,4.,noise_variance)
    
    # test_obj = hpx.VariationalRegularization(msh,Vh,misfit_form_reg)   
    # # print(test_obj.cost(x[hpx.PARAMETER]))
    
    # #trying to do the above
    # mfun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER])
    # loc_cost = misfit_form_reg(mfun)
    # glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))

    # # print(loc_cost,'\n')
    # print(glb_cost_proc)

    # misfit_form = H1TikhonvFunctional(3.,4.,noise_variance)
    # misfit = hpx.VariationalRegularization(msh,Vh,misfit_form)

    # # print(rank,":",misfit.cost(x[hpx.PARAMETER]))
    # out = pde.generate_parameter()
    # misfit.grad(x[hpx.PARAMETER],out)

    # print(out.array.min(),":",out.array.max())

    # test_func = hpx.vector2Function(out,Vh[hpx.PARAMETER])

    #have plots to compare to when you modify:
    # #True function plot - correct
    # with dlx.io.XDMFFile(msh.comm, "VAR_REG_MISFIT_GRAD_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(test_func)




    #compare loc_cost to one from misfit.py
    # misfit_form_misfit = PACTMisfitForm(d, noise_variance)
    # u_fun = hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE])
    # m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER])
    # loc_cost = misfit_form_misfit(u_fun,m_fun)
    
    # print(loc_cost)


    # glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
    # print(glb_cost_proc)
    # mfun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER])
    
#     #Fixing methods from PDEProblem.py class
#     #1. evalGradientParameter
#     #Current version
#     # test_val = pde.evalGradientParameter(x)
    
#     #Desired version
#     # test_val = pde.generate_parameter()
#     # pde.evalGradientParameter(x,test_val)
    
#     #2. solveAdj
#     #Current version - modified and works

#     #Checking methods from prior.py
#     # PRIOR
#     prior_mean = dlx.fem.Function(Vh_m)
#     # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.01))
#     prior_mean.x.array[:] = 0.01
#     # prior_mean.x.scatter_forward() - not needed
#     prior_mean = prior_mean.x


#     prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"],mean =  prior_mean)

#     #testing the methods in prior
#     #init works, sample- works (same in serial and parallel), cost works (serial, parallel), grad works (serial, parallel)

#     model = hpx.Model(pde, prior, misfit)

#     # noise = prior.init_vector("noise")
#     # m0 = prior.init_vector(0)    
#     # noise.array[:] = 3.
#     # prior.sample(noise,m0)

#     noise = prior.generate_parameter("noise")
#     m0 = prior.generate_parameter(0)    
#     noise.array[:] = 3.
#     prior.sample(noise,m0)

#     print(rank,":",m0.array.min(),":",m0.array.max())
#     #creating plot to test against after modifying prior.init_vector
#     test_func = hpx.vector2Function(m0,Vh[hpx.PARAMETER])

#     #have plots to compare to when you modify:
#     #True function plot - correct
#     # with dlx.io.XDMFFile(msh.comm, "CURRENT_m0_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
#     #     file.write_mesh(msh)
#     #     file.write_function(test_func)




#     #dummy example for non-zero values in x[STATE] after solveFwd
#     m0 = dlx.fem.Function(Vh_m) 
#     m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
#     m0.x.scatter_forward() 
#     m0 = m0.x

#     h = model.generate_vector(hpx.PARAMETER)
#     # rng = hpx.parRandom(comm)
#     # rng.normal(1., h)

#     hpx.parRandom(comm).normal(1.,h)

#     # print(rank,":",h.array.min(),":",h.array.max())
#     # eps, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=False,verbose=(rank == 0),eps=None)
    
#     # if(rank == 0):
#     #     print(err_grad)
#     # print(f'hello from rank {comm.rank}')

# #####################################
#     # #using class Variational Regularization
#     # #letting m0 = 3.
#     # new_prior = hpx.VariationalRegularization(msh,Vh, H1TikhonvFunctional(prior_param['gamma'],prior_param['delta'],3.))
#     # #test methods of this prior
#     # m0 = dlx.fem.Function(Vh_m) 
#     # m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
#     # m0.x.scatter_forward() 
#     # m0 = m0.x
#     # print(rank,":",new_prior.cost(m0))    
# #####################################

#     #PRNG:
#     #creating an example vector:
#     # d = dlx.fem.Function(Vh[hpx.STATE])
#     # expr = u_fun * ufl.exp(m_fun)
#     # hpx.projection(expr,d)
#     # d.x.scatter_forward()
    
#     h = model.generate_vector(hpx.PARAMETER) #dlx.la.Vector
#     # print(h.)
    
#     # # hp.parRandom.normal(1., h)
#     # dofmap = h.dofmap

#     # 1. Obtain the index map from the dofmap
#     index_map = h.index_map

#     # # 2. Generate random perturbation values (assuming uniformly distributed random values between -0.5 and 0.5)
#     num_dofs = index_map.size_global
#     random_perturbations = np.random.rand(num_dofs) - 0.5

#     # # 3. Update the values in the vector using the index map and random perturbations
#     # # Get local indices for this process
#     # local_indices = index_map.local_indices(False)

#     # help(dlx.cpp.common.IndexMap)
    
#     # print(index_map.local_range)
#     loc_range = h.index_map.local_range
#     loc_indices = np.arange(loc_range[0],loc_range[1])

###################################################################


    # print(rank,":",len(h.array),loc_range)
    # h.array = np.random.random(len(h.index_map.local_range))
    # print(h.index_map.local_range)

    #Attempt 2
    # h_petsc = dlx.la.create_petsc_vector_wrap(h)
    # loc_num_vals = len(h_petsc.getArray())
    # loc_arr = np.ones(loc_num_vals)
    # # print(h_petsc.getLocalVector())
    # # h_petsc.placeArray(loc_arr)
    # dlx.la.create_petsc_vector_wrap(h).placeArray(loc_arr)
    # print(h_petsc.array)
    # print(h.array)
    ###########################

    #Attempt 3
    # imap = h.index_map
    # # print(imap.size_local, imap.num_ghosts)

    # # h.array[:] = np.ones(imap.size_local + imap.num_ghosts)
    # num_local_values = imap.size_local + imap.num_ghosts
    
    # master_seed = 123
    # seed_sequence = np.random.SeedSequence(master_seed)
    
    # #Assigning seeds to each process
    # child_seeds = seed_sequence.spawn(nproc)
    # rng = np.random.MT19937(child_seeds[rank])
    
    # loc_random_numbers = np.random.default_rng(rng).normal(loc=0,scale= np.sqrt(noise_variance),size=num_local_values)
    # h.array[:] = loc_random_numbers
    # dlx.la.create_petsc_vector_wrap(h).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
            
    # print(rank,":",h.array.min(),":",h.array.max())

        


 # print(rank,":",h.array.min(),":",h.array.max())

    # # Get global indices for this process
    # global_indices = index_map.local_to_global(local_indices)

    # # Assign random perturbation values to the corresponding DOFs in the vector
    # vector_values = h.get_local()
    # vector_values[local_indices] += random_perturbations
    # h.set_local(vector_values)

    # # Ensure the changes are synchronized across processes
    # h.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=dolfinx.cpp.la.Vector.Mode.FAST)

    # test_obj = dlx.fem.Function(Vh_m)
    # test_obj.x.array[:] = 0.01

    # print(rank,":",test_obj.x.array[:])


    # test_obj.interpolate(lambda x: np.full((x.shape[1],),0.01))
    # test_obj.x.scatter_forward()
    # test_obj = test_obj.x


    # print(rank,":",test_obj.x.array)
    # test_obj = dlx.fem.Function(Vh_m)
    # test_obj.interpolate(lambda x: 0.01)
    # test_obj.x.scatter_forward()

    # prior_mean.interpolate(lambda x: 0.01)

    # model = hpx.Model(pde, new_prior, misfit)


    # mfun = hpx.vector2Function(m0,Vh[hpx.PARAMETER])
    # loc_cost = new_prior.functional_handler(mfun)
    # print(type(loc_cost))
    # # loc_cost = dlx.fem.form(loc_cost)
    # # glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
    # # return self.mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )


    # prior_mean = dlx.fem.Function(Vh_m)
    # prior_mean.x.array[:] = 0.01


    # prior_mean.interpolate(dlx.fem.Constant(msh,(0.01,0.01)))

    # print(rank,":",len(prior_mean.x.array))

    # prior_mean.interpolate(lambda x: np.full((x.shape[1],),0.01))
    # prior_mean.x.array[:] = 0.01
    # prior_mean.x.scatter_forward() - not needed
    # prior_mean.interpolate(dlx.fem.Constant(msh,0.01))
    # prior_mean = prior_mean.x

    # print(rank,":",prior_mean.array[:])

    # test_obj = dlx.fem.Constant(msh,0.01)
    # prior_mean.interpolate()
    # prior_mean.interpolate(lambda x: np.full(x.shape[1],),0.01)

    # prior_mean.interpolate(lambda x: np.full((x.shape[1],), 0.01))

    # prior_mean.interpolate(lambda x: np.full((x.shape[1],), dlx.fem.Constant(msh,0.01)))


if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)
