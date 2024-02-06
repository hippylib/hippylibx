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

# from NonGaussianContinuousMisfit import NonGaussianContinuousMisfit


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

    test_obj =  dlx.la.vector(Vh[hpx.STATE].dofmap.index_map, Vh[hpx.STATE].dofmap.index_map_bs)

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

    #to preserve values after using solveFwd
    # m_fun_true = dlx.fem.Function(Vh_m)
    # m_fun_true.x.array[:] = m_true.x.array[:]

    # m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>
    # print(type(m_true))

    # m_true = m_true.x #<class 'dolfinx.cpp.la.Vector_float64'>
    # print(type(m_true))


    u_true = pde.generate_state()   #a vector, not a function, <class 'dolfinx.la.Vector'>
    
    x_true = [u_true, m_true, None]  #list of dlx.la.vectors    

    #how to get comm from dlx.la.Vector object??
    # u_true_petsc = hpx.create_petsc_vector_wrap(u_true,comm)
    
    #is .x from a function and .vector - 
    #  local to the procecss??
    # print(type(m_true))
    
    # print(rank,":",len(m_true.x.array[:])) #909, 885, 910, 917, total = 3621
    # print(rank,":",m_true.vector.getLocalSize()) #818, 816, 831, 822, total = 3287

    #goal: m_true -> a dlx.fem.Function object, how to get the dlx.la.Vector object
    #to convert to petsc4py Vec for solving? 
    #.x gives a dlx.cpp.la.Vector_float64 object. Can  that be sent to wrap function??
    #Ans: Yes, it works.
    #1. create to petsc vec -> use wrap function
    #2. convert to 

    # m_true_petsc = hpx.create_petsc_vector_wrap(m_true.x,comm) #works.

    # print(type(m_true_petsc))

    # print(rank,":",m_true_petsc.getArray().min(),":",m_true_petsc.getArray().max())
    


    # print(u_true_func.getArray().min(),":",u_true_func.getArray().max())

    # print(type(test_obj),type(m_true)) 
    # print(type(u_true))

    # u_true_petsc:
    # print(u_true.)
    # u_map = u_true.map
    # ghosts = u_map.ghosts.astype(petsc4py.PETSc.IntType)
    # bs = u_true.bs
    # u_size = (u_map.size_local *bs, u_map.size_global*bs)

    # u_true_petsc = petsc4py.PETSc.Vec().createGhostWithArray(ghosts, u_true.array, size=u_size, bsize=bs, comm = comm)


    # print(type(u_true_petsc))

    # print(u_true_petsc.getArray().min(),":",u_true_petsc.getArray().max())

    # u_true_petsc_vec = dlx.la.create_petsc_vector_wrap(u_true)
    # def create_petsc_vector_wrap(x: Vector):
    # help(dlx.la)


    #              PETSc.Vec().createGhostWithArray(ghosts, x.array, size=size, bsize=bs, comm=map.comm)  # type: ignore

    #trying to convert a dlx.la.Vector object to a pesc4py Vec object that 
    #can be use d in solver.


    # print(type(m_true))
    # test_map = m_true.bs

    # print(m_true.map) #works

    # m_true_petsc = dlx.la.create_petsc_vector_wrap(test_obj)

    # print(dlx.__version__)

    # m_true_petsc = hpx.create_petsc_vector_wrap(m_true.x, comm)


    #function -> .x and .vector give dlx.cpp.la.Vector64 and 

    # m_true_func_from_petsc = hpx.vector2Function(m_true_petsc, Vh[hpx.PARAMETER])

    # print(type(x_true[hpx.PARAMETER]))
    #creating a petsc vector wrap of x_true[hpx.PARAMETER] -> dlx.la.Vector

    # test_obj = hpx.create_petsc_vector_wrap(x_true[hpx.PARAMETER],comm)

    # test_obj = dlx.la.create_petsc_vector_wrap(x_true[hpx.PARAMETER])
    # print(type(test_obj))

#############################################################

    pde.solveFwd(u_true, x_true) 
    

    x_true[hpx.STATE] = u_true
    adj_vec = pde.generate_state()
    #a test vector
    # adj_rhs = pde.generate_state()

    #example case to check working of solveAdj in serial and parallel.
    adj_rhs = dlx.fem.Function(Vh_m)
    adj_rhs.interpolate(lambda x: np.log(0.34) + 3.*( ( ( (x[0]-1.5)*(x[0]-1.5) + (x[1]-1.5)*(x[1]-1.5) ) < 0.75) )) # <class 'dolfinx.fem.function.Function'>
    adj_rhs.x.scatter_forward() 
    adj_rhs = dlx.la.create_petsc_vector_wrap(adj_rhs.x)

    pde.solveAdj(adj_vec, x_true, adj_rhs, comm)
    x_true[hpx.ADJOINT] = adj_vec

    # print(rank,":",adj_vec.array.min(),":",adj_vec.array.max())

    # grad_val = pde.evalGradientParameter(x_true)

    # print(type(grad_val)) #petsc4pyVec

    # print(rank,":",grad_val.getArray().min(),":",grad_val.getArray().max())


    ############################
    # LIKELIHOOD
    u_fun = hpx.vector2Function(x_true[hpx.STATE],Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER],Vh[hpx.PARAMETER])
    
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)

    d.x.scatter_forward()

    # print(rank,":",d.x.array.min(),":",d.x.array.max())

    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(msh, Vh, misfit_form)
    # grad = misfit.grad(0,x_true) #different plots in serial and parallel
    
    # print(rank,":",grad.getArray().min(),":",grad.getArray().max())
    prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"])

    model = hpx.Model(pde, prior, misfit)

    noise = prior.init_vector("noise")

    m0 = prior.init_vector(0)

    # noise.set(3.)
    noise.array[:] = 3.

    # print(rank,":",len(noise.array))

    prior.sample(noise,m0)


    # print(type(m0),type(noise))
    # print(type(prior.sqrtM))

    # noise_petsc = dlx.la.create_petsc_vector_wrap(noise)

    # print(rank,":",noise_petsc.getArray().min(),":",noise_petsc.getArray().max())

    # print(rank,":",len(noise_petsc.getArray()))

    # rhs = prior.sqrtM * noise_petsc

    # print(type(noise_petsc))

    # rhs = prior.sqrtM*dlx.la.create_petsc_vector_wrap(noise)


    # print(rank,":",m0.array.min(),":",m0.array.max())


################################
    
    #dummy example for non-zero values in x[STATE] after solveFwd
    m0 = dlx.fem.Function(Vh_m) 
    m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m0.x.scatter_forward() 
    m0 = m0.x


    # prior.sample(noise,m0)

    eps, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0),eps=None)
    # index = 2
    # h = model.generate_vector(hpx.PARAMETER)
    # # # print(type(h))
    # h.array[:] = 5
    # x = model.generate_vector()


    # x[hpx.PARAMETER] = m0 #dlx.la.Vector

    # model.solveFwd(x[hpx.STATE], x)
    # model.solveAdj(x[hpx.ADJOINT], x ,Vh[hpx.ADJOINT])
    
    # # print(rank,":",x[hpx.ADJOINT].array.min(),":",x[hpx.ADJOINT].array.max())
    
    # cx = model.cost(x)
    # # misfit_cost = model.misfit.cost(x)
    # # reg_cost = model.cost(x[hpx.PARAMETER])

    # # print(type(x[hpx.STATE]),type(x[hpx.PARAMETER]),type(x[hpx.ADJOINT]))
    # # reg_cost = model.cost(x)

    # _,grad_x = model.evalGradientParameter(x,misfit_only=True)
    # # print(rank,":",grad_x.array.min(),":",grad_x.array.max())

    # grad_xh = grad_x.dot( dlx.la.create_petsc_vector_wrap(h) )
    # # print(rank,":",grad_xh)

    # eps = None
    # if eps is None:
    #     n_eps = 32
    #     eps = np.power(.5, np.arange(n_eps))
    #     eps = eps[::-1]
    # else:
    #     n_eps = eps.shape[0]
    
    # err_grad = np.zeros(n_eps)
    # err_H = np.zeros(n_eps)
    
    # # print(type(x[hpx.STATE]),type(x[hpx.PARAMETER]), type(x[hpx.ADJOINT]),type(h),type(m0) )


    # # print(rank,":",m0.array.min(),":",m0.array.max())
    # # x_plus = model.generate_vector()

    # # dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).axpy(1.,dlx.la.create_petsc_vector_wrap(m0))
    
    # # dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    # # dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

    # # print(rank,":",x_plus[hpx.PARAMETER].array.min(),":",x_plus[hpx.PARAMETER].array.max())


    # for i in range(n_eps):
    #     my_eps = eps[i]
        
    #     x_plus = model.generate_vector()
    #     # x_plus[hpx.PARAMETER].axpy(1., m0 )

    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).axpy(1.,dlx.la.create_petsc_vector_wrap(m0))
    
    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

    #     # x_plus[hpx.PARAMETER].axpy(my_eps, h)

    #     # if(i == 0):
    #     #     # print(dc,grad_xh)
    #     #     print(rank,":",x_plus[hpx.PARAMETER].array.min(),":",x_plus[hpx.PARAMETER].array.max())

    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).axpy(my_eps,dlx.la.create_petsc_vector_wrap(h))
    
    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    #     dlx.la.create_petsc_vector_wrap(x_plus[hpx.PARAMETER]).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

    #     # if(i == 0):
    #     #     # print(dc,grad_xh)
    #     #     print(rank,":",x_plus[hpx.PARAMETER].array.min(),":",x_plus[hpx.PARAMETER].array.max())

    #     model.solveFwd(x_plus[hpx.STATE], x_plus)

    #     dc = model.cost(x_plus)[index] - cx[index]
                
    #     err_grad[i] = abs(dc/my_eps - grad_xh)
        
        # if(i == 0):
        #     # print(dc,grad_xh)
        #     print(rank,":",x_plus[hpx.STATE].array.min(),":",x_plus[hpx.STATE].array.max())


        # Check the Hessian
        # grad_xplus = model.generate_vector(hpx.PARAMETER)

    # if(rank == 0):
    #     print(err_grad)

    # ytHx = H.inner(yy,xx)
    # xtHy = H.inner(xx,yy)
    # if np.abs(ytHx + xtHy) > 0.: 
    #     rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    # else:
    #     rel_symm_error = abs(ytHx - xtHy)
    # if verbose:
    #     print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
    #     if rel_symm_error > 1e-10:
    #         print( "HESSIAN IS NOT SYMMETRIC!!")
            
    # return eps, err_grad, err_H



    # print(rank,":",misfit_cost)
    # print(rank,":",reg_cost)
    
    # print(rank,":",cx)


    # print(rank,":",x[hpx.PARAMETER].array.min(),":",x[hpx.PARAMETER].array.max())
    # model.solveAdj(x[ADJOINT], x)

    # m_fun_true = dlx.fem.Function(Vh[hpx.PARAMETER])
    # m_fun_true.x.array[:] = m0.x.array[:]

################################
    # print(type(m0))

    # m0.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    # m0.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
    
    # print(rank,":",len(m0.getArray()))

    # print(type(m0))
    
    # m0.setLGMap(Vh[hpx.PARAMETER].dofmap.index_map)
    
    # print(rank,":",m0.getLGMap())

    # model.solveFwd(x[hpx.STATE], x,comm)

    # mfun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER])

    # fun = dlx.fem.Function(Vh[hpx.PARAMETER])
    # test_object = Vh[hpx.PARAMETER].tabulate_dof_coordinates()
    # print(rank,":",m0.array.min(),":",m0.array.max())
########################################################
    # print(type(m0))

    # test_obj = prior.sqrtM #3287, 19206
    # ncols_1 = test_obj.getSize()[1]
    # nrows_1 = test_obj.getSize()[0]

    # #need to make a dlx.la.Vector type object given a certain size.
    # t2 = pde.generate_parameter()
    # # print(t2.index_map.size_local)
    # # print(type(t2))

    # test_obj2 = prior.A #3287, 3287
    # ncols_2 = test_obj2.getSize()[1]
    # nrows_2 = test_obj2.getSize()[0]

    # # print(nrows_1,ncols_1)
    # # print(nrows_2,ncols_2)
    
    # m_length = ncols_2
    
    # # m0 = dlx.la.Vector(#index_map that describes size and distribution of Vector)
    # # m0 = dlx.la.Vector(m_true.index_map)
    # # print(type(m_true.index_map))

    # custom_index_map = dlx.cpp.common.IndexMap(comm, ncols_2)
    
    # m0_cust_2 = dlx.la.vector(custom_index_map) #dolfinx.la.Vector

    # # print(custom_index_map,custom_index_map.local_to_global)    

    # m0_cust_3 = dlx.la.vector(custom_index_map) #dolfinx.la.Vector
    
    # # print(Vh_phi._ufl_element.degree())    

    # # print(Vh_phi.num_sub_spaces)


    # qdegree = 2*Vh_phi._ufl_element.degree()

    # element = ufl.FiniteElement("Quadrature", Vh_phi.mesh.ufl_cell(), qdegree, quad_scheme="default")

    # Vh_test = dlx.fem.FunctionSpace(msh, element) 
    
    # # print(Vh_test.dofmap.index_map)
    # m_test = dlx.la.vector(Vh_test.dofmap.index_map)
    # print(rank,":",len(m_test.array))

    # help(dlx.fem.FunctionSpace)

    # print(Vh_test.)

    # m_test = dlx.fem.Function(Vh_test) 
    # m_test.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    

    # m_test.x.scatter_forward() 
    # m_test = m_test.x

    # print(rank,":",len(m_test.array))





    #testing global distribution:
    # m0_cust_2 = dolfinx.la.createVectorGlobal(ncols_1, comm)
    # custom_index_map_2 = m_true.index_map

    # m0_cust_3 = dlx.la.vector(custom_index_map_2) #dolfinx.la.Vector    
    # print(rank,":",len(m0_cust_3.array))

    #get the comm assc with Vh passed to prior
    # print(type(Vh_phi))
    # help(dolfinx.fem.function.FunctionSpaceBase)
    # print(Vh_phi.mesh.comm)


    # m0_cust_2.x = 2.

    # local_size = m0_cust.local_size

    # local_indices = m0_cust.local_indices

    # print(rank,":",local_size,":")

    # m0_cust.array[:] = 5.
    # print(type(m_true))
    # m_true_petsc = dlx.la.create_petsc_vector_wrap(m_true) #works
    # m_true_petsc = dlx.la.create_petsc_vector_wrap(m0_cust) #doesn't work

    # print(type(m_true),type(m0_cust_2))

    # print(rank,":",m_true.array.min(),":",m_true.array.max())            
    # m_true.x = 3.
    # print(type(m_true.x))

    # help(dlx.la.Vector)
    # print(rank,":",m_true.array.min(),":",m_true.array.max())            
    
    # print(rank,":",m0_cust_2.array.min(),":",m0_cust_2.array.max())            

    # print(type(m0_cust_2))
    # print(rank,":",m0_cust_2.min_local,":",m0_cust_2.max_local)            

    # print(type(m0_cust_2))

    # m0_petsc_vec = dlx.la.create_petsc_vector_wrap(m0_cust)

    # m0_petsc_vec = dlx.la.create_petsc_vector_wrap(m_true)

    # print(type(m0))

    # print(rank,":",len(m0.array))
    # print(len(m_true.array))
    

    # print(rank,":",len(x[hpx.PARAMETER].array))

    #x[hpx.PARAMETER] has to be modified to have the same mapping as Vh[hpx.PARAMETER] function space
    # m0_mod = dlx.la.create_petsc_vector(Vh[hpx.PARAMETER].dofmap.index_map, Vh[hpx.STATE].dofmap.index_map_bs) 
    # test_obj = dlx.la.create_petsc_vector_wrap(x[hpx.PARAMETER])
    

    # print(type(x[hpx.PARAMETER]))

    #take values in m0

    # fun.x.array[:] = x[hpx.PARAMETER].array[:]

    # fun.x.array = x[hpx.PARAMETER].array


    # model.solveAdj(x[hpx.ADJOINT], x ,Vh[hpx.ADJOINT],comm)
    # rhs = model.misfit.grad(hpx.STATE, x)
    # model.problem.solveAdj(x[hpx.ADJOINT],x,rhs,comm)    
    # cx = model.cost(x)


    ############################
    







    




    # print(rank,":",adj_vec.array.min(),":",adj_vec.array.max())

    
    # print(rank,":",x_true[hpx.STATE].array.min(),":",x_true[hpx.STATE].array.max())
    # print(rank,":",x_true[hpx.PARAMETER].array.min(),":",x_true[hpx.PARAMETER].array.max())
    # print(rank,":",x_true[hpx.ADJOINT].array.min(),":",x_true[hpx.ADJOINT].array.max())
        
    

    # print(rank,":",u_true.array.min(),":",u_true.array.max())
    # print(rank,":",m_true.array.min(),":",m_true.array.max())
    
    # print(type(u_true))

    # p_true_func = hpx.vector2Function(x_true[hpx.ADJOINT],Vh[hpx.ADJOINT])

    #to see if create_petsc_vector_wrap has the correct values, do this:
    # func.x -> convert back to func, plot
    # func.vector -> convert back to func, plot
    # pass either or both of .x, .vector through petsc4pyVec, convert to func and plot

    #doing above for m_true

    #True function plot - correct
    # with dlx.io.XDMFFile(msh.comm, "P_true_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(p_true_func)


    # convert .x to a func and see plot? - correct
    # print(rank,":",m_true.x.array.min(),":",m_true.x.array.max())    
    # x_func = dlx.fem.Function(Vh[hpx.PARAMETER])
    # x_func.x.array[:] = m_true.x.array[:]
    # with dlx.io.XDMFFile(msh.comm, "X_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(x_func)

    #convert .vector to func and see plot - correct
    # vec_func = dlx.fem.Function(Vh[hpx.PARAMETER])
    # # vec_func.x.array[:] = m_true.vector.array #cannot do, different sizes.
    # vec_func.vector.array = m_true.vector.array 
    # with dlx.io.XDMFFile(msh.comm, "Vec_func_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(vec_func)
    

    #passing .x and .vector through create_petsc_wrap -> function and then plotting
    
    # x_petsc_vec = hpx.create_petsc_vector_wrap(m_true.x,comm)
    # x_petsc_func = dlx.fem.Function(Vh[hpx.PARAMETER])
    
    # print(rank,":",len(x_petsc_vec.getArray()),len(x_petsc_func.x.array)) #different lengths

    # x_petsc_func.vector.axpy(1.,x_petsc_vec)

    # print(rank,":",len(x_petsc_func.x.array))
    # print(rank,":",len(x_petsc_func.vector.array))

    # print(type(x_petsc_vec))
   
    # print(rank,":",len(x_petsc_func.vector),len(x_petsc_vec.getArray()))


    #2 objects:
    # m_true_x = m_true.x
    # m_true_vec = m_true.vector



    # print(rank,":",len(m_true_x.array),":",len(m_true_vec)    )


    #checking if solveFwd:
    #1. has the correct values in u_true
    #2. corrupted values in m_true.

    #1.
    # print(rank,":",u_true.array.min(),":",u_true.array.max())
    #convert u to a function and make plot, same for m_true to see if correct.
    # print(rank,":",len(u_true.array)) #total = 3621

    #testing the vector2function function
    
    # print(rank,":",m_true.array[:].min(),":",m_true.array[:].max())
    # test_func = dlx.fem.Function(Vh[hpx.PARAMETER])
    # test_func.x.array[:] = m_true.array[:]

    # m_true_func = hpx.vector2Function(m_true,Vh[hpx.PARAMETER])

    # print(rank,":",m_true_func.x.array[:].min(),":",m_true_func.x.array[:].max())

    # print(rank,":",len(test_func.x.array)) #works
    # print(rank,":",test_func.x.index_map) #works
    # print(rank,":",Vh[hpx.PARAMETER].dofmap.index_map)

    # # #to preserve values in x_true        
    # x_true[hpx.STATE] = u_true
    # x_true[hpx.PARAMETER] = m_fun_true.vector

    # m_true = x_true[hpx.PARAMETER]

    # adj_vec = pde.generate_state()
    # #a test vector
    # adj_rhs = pde.generate_state()
    
    # adj_rhs = dlx.fem.Function(Vh_m)
    # adj_rhs.interpolate(lambda x: np.log(0.34) + 3.*( ( ( (x[0]-1.5)*(x[0]-1.5) + (x[1]-1.5)*(x[1]-1.5) ) < 0.75) )) # <class 'dolfinx.fem.function.Function'>
    # adj_rhs.x.scatter_forward() 
    
    # ##############################   
    # pde.solveAdj_2(adj_vec, x_true, adj_rhs)
    # x_true[hpx.ADJOINT] = adj_vec
    # ##############################


    # adj_vec_func = hpx.vector2Function(adj_vec,Vh[hpx.ADJOINT])   
    # grad_val = pde.evalGradientParameter(x_true)
    # #same values in serial and parallel
    
    # grad_func = hpx.vector2Function(grad_val,Vh[hpx.STATE])
    
    # u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    # m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    
    # # # LIKELIHOOD
    # d = dlx.fem.Function(Vh[hpx.STATE])
    # expr = u_fun * ufl.exp(m_fun)
    # hpx.projection(expr,d)


    # hpx.parRandom(comm, 1., d.vector)
    
    # hpx.parRandom(comm, noise_variance, d.vector)
    # print(type(d))

    # print(rank,":",d.x.array[:].min(),":",d.x.array[:].max())

    # print(type(d.x)) #dolfinx.cpp.la.Vector_float64
    # print(type(d.x.array[:])) #numpy.ndarray
    # print(type(d.vector)) #petsc4py.PETSc.Vec
    # print(type(d.vector.getArray())) #numpy.ndarray




if __name__ == "__main__":    
  nx = 64
  ny = 64
  noise_variance = 1e-6
  prior_param = {"gamma": 0.05, "delta": 1.}

  run_inversion(nx, ny, noise_variance, prior_param)
