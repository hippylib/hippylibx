

import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os
import matplotlib

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

    pde.solveFwd(u_true, x_true, comm) #petsc4py.PETSc.Vec

########################################################
    
    x_true[hpx.STATE] = u_true
    adj_vec = pde.generate_state()
    #a test vector
    # adj_rhs = pde.generate_state()

    #example case to check working of solveAdj in serial and parallel.
    adj_rhs = dlx.fem.Function(Vh_m)
    adj_rhs.interpolate(lambda x: np.log(0.34) + 3.*( ( ( (x[0]-1.5)*(x[0]-1.5) + (x[1]-1.5)*(x[1]-1.5) ) < 0.75) )) # <class 'dolfinx.fem.function.Function'>
    adj_rhs.x.scatter_forward() 
    adj_rhs = hpx.create_petsc_vector_wrap(adj_rhs.x,comm)

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

    noise.set(3.)

    prior.sample(noise,m0)

    # # _, err_grad, _ = hpx.modelVerify(Vh,comm,model,m0,is_quadratic=False,misfit_only=True,verbose=(rank == 0),eps=None)
    index = 2
    h = model.generate_vector(hpx.PARAMETER)
    # print(type(h))
    h.array[:] = 5
    x = model.generate_vector()
    x[hpx.PARAMETER] = m0 #petsc4py.PETSc.Vec



    # m0.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    # m0.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
    
    # print(rank,":",len(m0.getArray()))

    # print(type(m0))
    
    # m0.setLGMap(Vh[hpx.PARAMETER].dofmap.index_map)
    
    # print(rank,":",m0.getLGMap())

    # model.solveFwd(x[hpx.STATE], x,comm)

    # mfun = hpx.vector2Function(x[hpx.PARAMETER],Vh[hpx.PARAMETER])

    fun = dlx.fem.Function(Vh[hpx.PARAMETER])
    test_object = Vh[hpx.PARAMETER].tabulate_dof_coordinates()
    
    # print(rank,":",m0.array.min(),":",m0.array.max())
########################################################
    
    # print(rank,":",len(x[hpx.PARAMETER].array))

    #x[hpx.PARAMETER] has to be modified to have the same mapping as Vh[hpx.PARAMETER] function space
    # m0_mod = dlx.la.create_petsc_vector(Vh[hpx.PARAMETER].dofmap.index_map, Vh[hpx.STATE].dofmap.index_map_bs) 
    # test_obj = hpx.create_petsc_vector_wrap(x[hpx.PARAMETER],comm)
    
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
