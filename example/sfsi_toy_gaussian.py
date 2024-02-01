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

from NonGaussianContinuousMisfit import NonGaussianContinuousMisfit

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


    #to preserve values after using solveFwd
    m_fun_true = dlx.fem.Function(Vh_m)
    m_fun_true.x.array[:] = m_true.x.array[:]

    m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>
    
    # print(m_true.min())
    # print(m_true.max())
    
    u_true = pde.generate_state()   #a vector, not a function, <class 'petsc4py.PETSc.Vec'>
    x_true = [u_true, m_true, None]     #list of petsc vectors    
    A_mat = pde.solveFwd(u_true, x_true) #petsc4py.PETSc.Vec

    # u_true_func = hpx.vector2Function(u_true,Vh[hpx.STATE])
    # print(rank,":",u_true_func.x.array[:].min(),":",u_true_func.x.array[:].max())

    # print(u_true.min(),":",u_true.max()) #same as fenics

    # print(m_true.min())
    # print(m_true.max())
    
    # #to preserve values in x_true        
    x_true[hpx.STATE] = u_true
    x_true[hpx.PARAMETER] = m_fun_true.vector

    m_true = x_true[hpx.PARAMETER]

    adj_vec = pde.generate_state()
    #a test vector
    adj_rhs = pde.generate_state()

    # hpx.random.parRandom(comm,np.sqrt(noise_variance),adj_rhs) #fix this so it uses SeedSequence    
    #adj_rhs needs to consistantly have the same values when using different number of 
    #processes, so interpolate using expressions
    
    adj_rhs = dlx.fem.Function(Vh_m)
    adj_rhs.interpolate(lambda x: np.log(0.34) + 3.*( ( ( (x[0]-1.5)*(x[0]-1.5) + (x[1]-1.5)*(x[1]-1.5) ) < 0.75) )) # <class 'dolfinx.fem.function.Function'>
    adj_rhs.x.scatter_forward() 

    # print(rank,":",adj_rhs.x.array.min(),":",adj_rhs.x.array.max())

    # adj_rhs = adj_rhs.vector
    # print(rank,":",adj_rhs.vector.min(),":",adj_rhs.vector.max())            
    # print(adj_rhs.min(),":",adj_rhs.max())
    # print(rank,":",adj_vec.min(),":",adj_vec.max())

    # print(rank,":",adj_rhs.vector.min(),":",adj_rhs.vector.max())            
    # print(rank,":",adj_vec.min(),":",adj_vec.max())            
    
    ##############################   
    pde.solveAdj_2(adj_vec, x_true, adj_rhs)
    x_true[hpx.ADJOINT] = adj_vec

    # print(rank,":",adj_rhs.x.array.min(),":",adj_rhs.x.array.max())

    # print(rank,":",adj_vec.min(),":",adj_vec.max())            
    ##############################

    ##############################
    # adj_rhs = adj_rhs.vector
    # adj_vec_2 = pde.generate_state()

    # print(rank,":",adj_rhs.min(),":",adj_rhs.max())            
    # print(x_true[hpx.STATE].min(),":",x_true[hpx.STATE].max())
    # print(x_true[hpx.PARAMETER].min(),":",x_true[hpx.PARAMETER].max())
    # pde.solveAdj(adj_vec_2, x_true, adj_rhs)
    
    # print(rank,":",adj_vec_2.min(),":",adj_vec_2.max())            



    # x_true[hpx.ADJOINT] = adj_vec

    # print(rank,":",adj_vec.min(),":",adj_vec.max())            
    
    ##############################

    # u = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    # m = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    # p = dlx.fem.Function(Vh[hpx.ADJOINT])
    # du = ufl.TestFunction(Vh[hpx.STATE])
    # dp = ufl.TrialFunction(Vh[hpx.ADJOINT])
    # varf = pde_handler(u, m, p)
    # adj_form = ufl.derivative( ufl.derivative(varf, u, du), p, dp )
    # print(adj_rhs.min(),":",adj_rhs.max()) #-1.0788096613719298, 1.9211903386280702
    # Aadj = dlx.fem.petsc.assemble_matrix(dlx.fem.form(adj_form),bcs = [])
    # print(adj_rhs.min(),":",adj_rhs.max()) #-3.420763640362111e+306, 1.1652105010162572e+301
    # Aadj.assemble()
   
    # self.solver.setOperators(Aadj)
    
    # self.solver.solve(adj_rhs, adj)

    # print(rank,":",adj_rhs.vector.min(),":",adj_rhs.vector.max())            
    
    # print(rank,":",adj_vec.min(),":",adj_vec.max())
    
    # print(type(adj_vec))
    # print(rank,":",adj_rhs.min(),":",adj_rhs.max())
    # print(rank,":",adj_vec.min(),":",adj_vec.max())
    # print(rank,":",adj_vec.getLocalSize())

    #gather the adj_vec computed by each process to 0 and see if that vector is
    #same as the vector generated using mpirun -n 1

    #gathering the adj_vec to process 0:
    # adj_result = comm.gather(adj_vec,0 )
    #plots for adj_vec - in serial and parallel

    adj_vec_func = hpx.vector2Function(adj_vec,Vh[hpx.ADJOINT])   

    # print(x_true[hpx.STATE].min(),":",x_true[hpx.STATE].max())
    # print(x_true[hpx.PARAMETER].min(),":",x_true[hpx.PARAMETER].max())
    # print(x_true[hpx.ADJOINT].min(),":",x_true[hpx.ADJOINT].max())

    grad_val = pde.evalGradientParameter(x_true)
    
    # print(grad_val.getLocalSize())
    # print(grad_val.min(),":",grad_val.max())
    # adj_vec_func.x.scatter_forward()  
    # print(rank,":",adj_vec_func.x.array[:].min(),":",adj_vec_func.x.array[:].max())

    #same values in serial and parallel
    
    grad_func = hpx.vector2Function(grad_val,Vh[hpx.STATE])

    # print(rank,":",grad_func.x.array[:].min(),":",grad_func.x.array[:].max())

    # adj_func = hpx.vector2Function(x_true[hpx.ADJOINT],Vh[hpx.ADJOINT])
    
    # print(rank,":",adj_vec_func.x.array[:].min(),":",adj_vec_func.x.array[:].max())   

    # with dlx.io.XDMFFile(msh.comm, "attempt_p_true_vec_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(adj_func)

    # pde_grad = pde.evalGradientParameter(x_true)  

    # print(x_true[hpx.STATE].min(),":",x_true[hpx.STATE].max())
    # print(x_true[hpx.PARAMETER].min(),":",x_true[hpx.PARAMETER].max())

    # grad = pde.evalGradientParameter(x_true)
    # print(rank,":",grad.min(),":",grad.max())

    # print(m_true.min(),":",m_true.max()) #same as fenics

    # print(m_true.min())
    # print(m_true.max())
    
    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    
    # # LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)
    
    # hpx.random.parRandom(comm,np.sqrt(noise_variance),d) #fix this so it uses SeedSequence    
    # print(rank,":",d.x.array[:])
    # print(d.x.array[:].min(),":",d.x.array[:].max()) #same as fenics
    # d.x.scatter_forward()
    # print(d.x.array[:].min(),":",d.x.array[:].max()) #same as fenics
    #values from normal dist have to be added to d
    
    #comment out the random perturbation to see if you get same values in serial and parallel runnning
    #with multiple processes.
    # hpx.random.parRandom(comm,np.sqrt(noise_variance),d.vector) #fix this so it uses SeedSequence    
    
    # print(rank,":",d.x.array[:])
    
    d.x.scatter_forward()
    
    # print(d.x.array[:].min(),":",d.x.array[:].max()) #same as fenics
    # print(rank,":",d.x.array.min(),":",d.x.array.max())

    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    # # PRIOR
    grad = misfit.grad(0,x_true) #different plots in serial and parallel
    # print(grad.getArray().min(),":",grad.getArray().max())

    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])

    # ones = dl.interpolate(one_constant, Qh).vector()

    # cannot implement prior as class _Prior uses Multivector which derives from 
    # a cpp_module.Multivector c++ something. So, this won't work. Try to use the
    # methods from the Regularization.py file to move ahead with the modelVerify.py

    #dummy, haven't created a prior.

    # prior = pde.generate_state()
    # prior = hpx.test_prior()
    # m0 = prior.init_vector(0)

    # model = hpx.Model(pde,prior,misfit)

    # hpx.modelVerify(model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))

    trial = ufl.TrialFunction(Vh_phi)
    test  = ufl.TestFunction(Vh_phi)
    varfM = ufl.inner(trial,test)*ufl.dx       
    M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM)) #used as set_Operator, so must be matrix


    # Vh = Vh_phi #dolfinx.fem.function.FunctionSpace
    # test_obj = Vh._ufl_element.degree() #1
    # print(test_obj)
    qdegree = 2*Vh_phi._ufl_element.degree()
    metadata = {"quadrature_degree" : qdegree}

    # print(type(Vh))

    num_sub_spaces = Vh_phi.num_sub_spaces #0
    # print(num_sub_spaces)

    element = ufl.FiniteElement("Quadrature", Vh_phi.mesh.ufl_cell(), qdegree, quad_scheme="default")

    # element = ufl.VectorElement("Quadrature", Vh.mesh.ufl_cell(), qdegree, dim=num_sub_spaces, quad_scheme="default")
    # Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
    
    Qh = dlx.fem.FunctionSpace(Vh_phi.mesh, element)

    # Qh = dlx.fem.FunctionSpace(msh, ("CG",1))
    
    ph = ufl.TrialFunction(Qh)
    qh = ufl.TestFunction(Qh)
    Mqh = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,qh)*ufl.dx(metadata=metadata)) ) #petsc4py.PETSc.Mat

    # print(Mqh.getBlockSizes()) #(1,1)
    # print(Mqh.getLocalSize()) #(19206, 19206)
    
    ones = Mqh.createVecRight()
    ones.set(1.)
    dMqh = Mqh.createVecLeft()
    Mqh.assemble()
    Mqh.mult(ones,dMqh)

    # test_obj = Mqh.createVecLeft()
    # if(rank == 0):
    #     print(dMqh.getArray())

    dMqh.setArray(ones.getArray()/np.sqrt(dMqh.getArray()))
    
    # if(rank == 0):
    #     print(dMqh.getArray())

    # if(rank == 0):
    #     Mqh.getDiagonal(test_obj)
    #     print(test_obj.getArray())
    
    Mqh.setDiagonal(dMqh)
    
    # if(rank == 0):
    #     Mqh.getDiagonal(test_obj)
    #     print(test_obj.getArray())
    
    # if(rank == 0):
    #     print(dMqh.getArray())

    MixedM = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,test)*ufl.dx(metadata=metadata)))
    MixedM.assemble()

    # print(MixedM.getLocalSize())
    # print(Mqh.getLocalSize())
    
    # sqrtM = MixedM.matMult(Mqh)
    
    # sqrtM = MixedM.matMult(Mqh,None,None)

    sqrtM = MixedM.matMult(Mqh)
    sqrtM.assemble() #may not be needed

    # print(type(dMqh))

    # prior = hpx.BiLaplacianPrior(Vh_phi,3.,4.,5.)
    
    prior = hpx.BiLaplacianPrior(Vh_phi,prior_param["gamma"],prior_param["delta"])

    # test_obj = prior.A
    # test_obj = prior.A.getDenseArray()
    
    ########################################   
    # nrows, ncols = prior.M.getSize()

    # dense_array = np.zeros((nrows, ncols), dtype=float)        
    # row_indices, col_indices, csr_values = prior.M.getValuesCSR()

    # for i in range(nrows):
    #     start = row_indices[i]
    #     end = row_indices[i + 1]
    #     dense_array[i, col_indices[start:end]] = csr_values[start:end]
    # print(np.max(dense_array),np.min(dense_array))
    ########################################   
    
    model = hpx.Model(pde, prior, misfit)

    # model_obj = hpx.Model(pde, prior, misfit)

    # #a different model_obj to implement the modelVerify method from scratch here,
    # #created here as noise and m0 are perturbed in the following lines.
    # model_obj = hpx.Model(pde,prior,misfit)

    noise = prior.init_vector("noise")

    m0 = prior.init_vector(0)
    # print(m0.getArray().min(),":",m0.getArray().max())

    # # # noise_func 
    #values from normal distribution have to be inserted in noise
    # hpx.random.parRandom(comm,1.,noise) #fix this so it uses SeedSequence

    #set values in noise to some constant value
    noise.set(3.)

    # print(noise.getArray().min(),":",noise.getArray().max())
    # # # print(m0.getArray().min(),":",m0.getArray().max())
    # # # print(rank,":",m0.getArray())
    # # print(m0.getArray().min(),":",m0.getArray().max())

    
    ################################
    prior.sample(noise,m0)

    #modelVerify now:
    index = 2
    h = model.generate_vector(hpx.PARAMETER)
    
    #Fix h with a constant value or a function (interp) that is consistent across
    #runs of 1 or more processes.
    # hpx.parRandom(comm,1.,h)

    h.set(5.)

    # print(h.min(),":",h.max())

    x = model.generate_vector()

    # modify m0 so you can have non-zero vector solution of x[STATE]    
    m0 = dlx.fem.Function(Vh_m)
    m0.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    m0.x.scatter_forward() 
    # m0 = m0.vector
    # m0_dupl = m0

    
    # print(m0.min(),":",m0.max())
    
    m_fun_true = dlx.fem.Function(Vh_m)
    m_fun_true.x.array[:] = m0.x.array[:]

    # m_true = m_true.vector #<class 'petsc4py.PETSc.Vec'>
    m0 = m0.vector
    x[hpx.PARAMETER] = m0

    model.solveFwd(x[hpx.STATE], x)
    
    # print(x[hpx.STATE].min(),":",x[hpx.STATE].max())
    
    # print(m0.min(),":",m0.max()) #values are messed up here.
    
    # print(x[hpx.PARAMETER].min(),":",x[hpx.PARAMETER].max())
    
    x[hpx.PARAMETER] = m_fun_true.vector
    m0 = m_fun_true.vector
    # print(x[hpx.PARAMETER].min(),":",x[hpx.PARAMETER].max())

    # print(x[hpx.STATE].min(),":",x[hpx.STATE].max())
    # print(x[hpx.PARAMETER].min(),":",x[hpx.PARAMETER].max())

    rhs = model.problem.generate_state()
    
    # rhs = model.misfit.grad(hpx.STATE, x)
    
    ##########################
    u_fun = hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER])
    x_fun = [u_fun, m_fun]
    x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
    # dl.assemble(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i]), tensor=out )

    L = dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[hpx.STATE], x_test[hpx.STATE]))

    rhs = dlx.fem.petsc.create_vector(L)

    with rhs.localForm() as loc_grad:
        loc_grad.set(0)

    dlx.fem.petsc.assemble_vector(rhs,L)

    rhs.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    # rhs.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ##########################
    
    rhs.scale(-1.) #slight difference in values.

    # print(x[hpx.ADJOINT].min(),":",x[hpx.ADJOINT].max())
    
    ################commented out from here
    ####################################################
    rhs_func = hpx.vector2Function(rhs,Vh[hpx.ADJOINT])
    # model.problem.solveAdj(x[hpx.ADJOINT],x,rhs)
    model.problem.solveAdj_2(x[hpx.ADJOINT],x,rhs_func)
    
    # print(rank,":",x[hpx.ADJOINT].min(),":",x[hpx.ADJOINT].max())

    # print(x[hpx.STATE].min(),":",x[hpx.STATE].max())
    # print(x[hpx.PARAMETER].min(),":",x[hpx.PARAMETER].max())
    # print(x[hpx.ADJOINT].min(),":",x[hpx.ADJOINT].max())

    cx = model.cost(x)

    # print(rank,":",cx)    
    grad_x = model.generate_vector(hpx.PARAMETER)
    calc_val,grad_x = model.evalGradientParameter(x,misfit_only=True)

    # print(rank,":",calc_val,":",grad_x.min(),":",grad_x.max())
    grad_xh = grad_x.dot( h )

    # print(rank,":",grad_xh)
    eps = None

    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]
    
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)


    # x_plus = model.generate_vector()
    # x_plus[hpx.PARAMETER].axpy(1., m0 )
    # x_plus[hpx.PARAMETER].axpy(eps[0], h)

    # print(m0.min(),":",m0.max())

    # print(x_plus[hpx.PARAMETER].min(),":",x_plus[hpx.PARAMETER].max())

    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus = model.generate_vector()
        x_plus[hpx.PARAMETER].axpy(1., m0 )
        x_plus[hpx.PARAMETER].axpy(my_eps, h)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())    
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())
        
        model.solveFwd(x_plus[hpx.STATE], x_plus)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())

        # model.solveAdj(x_plus[ADJOINT], x_plus)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())        
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())
        
        dc = model.cost(x_plus)[index] - cx[index]
        
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        # if(i == 1):
            # print(i,":",rank,":",dc,":",my_eps,":",grad_xh,":",err_grad[i])
            # print(i,":",rank,":",my_eps)
            # print(i,":",rank,":",err_grad[i])
            
        # Check the Hessian
        # grad_xplus = model.generate_vector(hpx.PARAMETER)

    ####################################################

    if(rank == 0):
        print(err_grad)


    # print(rank,":",x_plus[hpx.PARAMETER].min(),":",x_plus[hpx.PARAMETER].max())

    # print(rhs.min(),":",rhs.max())
    # rhs_func = hpx.vector2Function(rhs,Vh[hpx.STATE])
    # rhs_func.x.scatter_forward()

    #check if m0 is correct for 1 and multiple processes - is correct
    # m0_func = hpx.vector2Function(m0,Vh[hpx.PARAMETER])
    # adjoint_func = hpx.vector2Function(x[hpx.ADJOINT],Vh[hpx.ADJOINT])
    
    # with dlx.io.XDMFFile(msh.comm, "attempt_plot_adj_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(adjoint_func)

    # print(m0.min(),":",m0.max())

    # m0_func = hpx.vector2Function(m0,Vh[hpx.PARAMETER])
    # m0_func.x.scatter_forward()

    # print(rank,":",m0.getArray().min(),":",m0.getArray().max())

    # print(rank,":",m0_func.x.array.min(),":",m0_func.x.array.max())
    ################################
    
    #implementing prior.sample here manually to check for same values in serial and
    #parallel case
    # mat_sqrtM = prior.sqrtM
    # # print(rank,":",mat_sqrtM.getLocalSize())
    # rhs = mat_sqrtM.createVecLeft()
    # # print(rank,":",rhs.getArray().min(),":",rhs.getArray().max())

    
    # # print(rank,":",rhs.getLocalSize())
    # mat_sqrtM.mult(noise,rhs)
    
    # op_Asolver = prior.Asolver
    
    # #need to check if prior.A has been distrbuted correctly to the processes,
    # #i.e. min,max values same in 1, 4 process run.

    # mat_A = prior.A
    # # print(mat_A.getLocalSize())

    # # print(np.min(mat_A.getValuesCSR()[2]),":",np.max(mat_A.getValuesCSR()[2]))


    # op_Asolver.solve(rhs,m0)


    # print(m0.getArray().min(),":",m0.getArray().max())


    # numpy_mat_A = mat_A.convert("dense")
    # numpy_mat_A.getDenseArray()
    
    # numpy_mat_A = mat_A.getArray()
    
    # print(type(numpy_mat_A))

    # print(numpy_mat_A.view())


    # print(numpy_mat_A.getLocalSize())

    # print(rank,":",A_mat.getLocalSize())

    # print(rank,":",mat_A.getLocalSize())

    # print(type(Asolver))p

    # print(rank,":",rhs.getArray().min(),":",rhs.getArray().max())

    # print(rank,":",m0.getArray().min(),":",m0.getArray().max())

    # # print(m0.getArray().min(),":",m0.getArray().max())
    # # print(m0.getLocalSize())
    # # print(rank,":",m0.getArray())

    # eps, err_grad, _ = hpx.modelVerify(comm, model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))

    # print(rank,":",err_grad,'\n')
    # print(err_grad)

    # eps,err_grad, _ = hpx.modelVerify(comm, model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))
    # print(eps,'\n')

    # if(rank == 0):
    #     print(err_grad)

    #####################################
    #modelVerify here - step by step to see where the results of serial diverge from those in parallel.
    # index = 2
    # # #create a dummy model object
    # model_obj = hpx.Model(pde,prior,misfit)

    # h = model_obj.generate_vector(hpx.PARAMETER)
    # hpx.parRandom(comm, 1., h) #supply comm to this method
    # # # print(h.min(),":",h.max())

    # x = model_obj.generate_vector()
    # x[hpx.PARAMETER] = m0
    # model_obj.solveFwd(x[hpx.STATE], x)


    # print(rank,":",x[hpx.STATE].min(),":",x[hpx.STATE].max())

    # #self implement adj method to see if the values calculated in model.solveadj
    # #are correct. This is because solveAdj in model passes rhs as a vector while
    # #in self implemented case, solveAdj needed rhs to be passed as a function 
    # #for correct values to be preserved in adj solution vector.
    # #same answers for solveAdj (takes in rhs vector) and solveAdj2 (takes in rhs function)
     

    # #model.solveAdj
    # rhs = model_obj.problem.generate_state() #petsc Vec
    # # # print(rhs.getLocalSize()) #3287
    # rhs = model_obj.misfit.grad(hpx.STATE, x) #different in serial and parallel.
    # rhs.scale(-1.)

    # print(rank,":",rhs.getArray().min(),":",rhs.getArray().max()) #-3787918.8718535625, 40569.35882166516

    #checking grad from pdeProblem.py
    #grad_pde = pde.evalGradientParameter(x_true)
    

    # print(grad_pde.getLocalSize())

    # print(rank,":",grad_pde.getArray().min(),":",grad_pde.getArray().max())

    # model_obj.problem.solveAdj(x[hpx.ADJOINT], x, rhs)
    # print(x[hpx.ADJOINT].min(),":",x[hpx.ADJOINT].max()) #-6989207.061176334, 312016.12408125534
    
    # # print(type(x[hpx.ADJOINT]))
    # # u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])

    # # test_adj_func = dlx.fem.Function(Vh[hpx.STATE])


    # # test_adj_func = hpx.vector2Function(x[hpx.ADJOINT], Vh[hpx.ADJOINT])
    # test_adj_ans = pde.generate_state()

    # #the values in test_adj_func.vector has to be same as rhs
    # # test_adj_func.vector[:] = rhs.getArray()
    
    # # print(rhs.getArray(),'\n')
    # # print(test_adj_func.vector[:])

    # # print(type(test_adj_func))

    # # test_adj_func.x.scatter_forward()

    # test_adj_func = hpx.vector2Function(rhs,Vh[hpx.ADJOINT])
    # pde.solveAdj_2(test_adj_ans,x,test_adj_func)

    # print(test_adj_ans.min(),":",test_adj_ans.max()) #-6989207.061176334, 312016.12408125534

    # if(rank == 0):

        # scale_val = err_grad[0]/eps[0]
        # second_val = [value*scale_val for value in eps]
        # plt.figure()
        # plt.subplot(121)
        # plt.loglog(eps, err_grad, "-ob", eps, second_val, "-.k")
        # plt.title("FD Gradient Check")
        # plt.show()

        # print(eps,'\n')
        # print(err_grad)
        # scale_val = err_grad[0]/eps[0]
        # second_val = [value*scale_val for value in eps]

        # plt.figure()
        # plt.subplot(121)
        # plt.loglog(eps, err_grad, "-ob", eps, second_val, "-.k")
        # plt.title("FD Gradient Check")
        # plt.show()

    # print(rank,":",err_grad,'\n')
    # print(err_grad)

    # print(eps,'\n')
    # print(err_grad)

    #make the log-log plot of the above 2 lists
    
    # if(rank == 0):
        # print("Hello")
        # plt.figure()
        # plt.subplot(121)
        # plt.show()
    
    # if rank == 0: 

    # plt.figure()
    
    # plt.subplot(121)
    # plt.show()


    # print(rank,":",noise.getArray())

    # self.R = _BilaplacianR(self.A, self.Msolver)      
    # self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
    
    # print(sqrtM.getLocalSize())
    
    # print(MixedM.getDiagonal())

    # print(type(MixedM))
    # print(type(dMqh))
    # print(rank,":",len(dMqh.getArray()))

    # Mqh.zero()

    # print(type(Mqh))

    # one_constant = dlx.fem.Constant(Vh.mesh, petsc4py.PETSc.ScalarType(1.))

    # print(type(Vh_phi))
    # print(type(Qh))
    
    #interpolate 1 over Qh - how to?    
    
    # print(Qh)

    # ones = dlx.fem.Function(Qh)
    # ones.interpolate(lambda x: np.ones(x.shape[1]))

    # ones.interpolate(lambda x: np.full((x.shape[1.) )

    # two_constant = dlx.fem.Constant( Vh.mesh, petsc4py.PETSc.ScalarType(tuple( [1.]*num_sub_spaces) ))
    # print(two_constant)

    # m_true = dlx.fem.Function(Vh_m)
    # m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m_true.x.scatter_forward() 

    # ones = dlx.fem.Function(Qh)
    # ones.interpolate(lambda x: np.full((x.shape[1],),1.) )

    # print(type(element))
    
    # print(type(Vh_phi))
    # print(type(Qh))

    # ones.sub(0).interpolate( one_constant, Vh )
    # ones.interpolate(lambda x: np.full((x.shape[1],),one_constant) )
    # ones.interpolate(lambda x: np.full((x.shape[1],),1.) )

    # ones.interpolate(lambda x: np.full((x.shape[1],),0.))

    # ones = dl.interpolate(one_constant, Qh).vector()
    # print(one_constant)

    # print(Vh.mesh.comm)

    # print(type(element))

    # test_obj = Vh.mesh
    # test_obj = Vh.mesh.ufl_cell #method
    # test_obj = Vh.mesh.ufl_cell() #ufl.cell.Cell

    #following 2 lines not needed
    # num_sub_spaces = Vh.element.num_sub_elements
    # print(num_sub_spaces)

    # model = model(pde,prior,misfit)

    #how to get the Vh assc with the petsc4py Vec object - for example u_true?
    #I need it to access the dofmap.index_map_ds to create a vector in the prior.py methods needed.

    #multiplying a petsc4py Mat with a petsc4py Vec
    # print(type(A_mat))
    # print(type(u_true))

    # print(type(A_mat))
    # print(A_mat.comm())
    # help1 = A_mat.createVecLeft()
    # print(len(help1.getArray()))
    # print(type(help1))

    # print(A_mat.getSizes())

    
    # dm = u_true.getDM()


    # dm = u_true.getDM()
    # fs = dlx.cpp.mesh.create_functionspace( dm, )

    # print(type(u_true))
    # print(u_fun.x.array.min(),":",u_fun.x.array.max())

    # print(rank,":",len(u_fun.x.array))

    # print(rank,":",len(u_true.getArray()))

    # test_obj = u_true.duplicate()
    # print(rank,":",len(test_obj.getArray()))

    # print(test_obj.getArray())

    # print(type(dm))    
    # fs = dm.getFunctionSpace()

    # test_obj = dlx.la.create_petsc_vector(Vh[hpx.STATE].dofmap.index_map, Vh[hpx.STATE].dofmap.index_map_bs) 
    # print(type(u_true))

    # print(type(u_true.comm)) #petsc4py.PETSc.Comm
    # print(type(comm)) #mpi4py.MPI.Intracomm


    # print(test_obj.comm)
    # print(test_obj)
    # model.py testing
    #1. need prior - only have Regularization.py


    # test_model = hpx.model(pde,  )


    #scaling a petsc4py Vec object
    # test_vec = pde.generate_state() #petsc4py.PETSc.Vec
    # test_vec.setArray(1.)
    # test_vec.scale(4.)
    # print(test_vec.getArray())
 

    #testing value preservation
    # print(rank,":",u_fun.x.array.min(),":",u_fun.x.array.max())
    # print(rank,":",m_fun.x.array.min(),":",m_fun.x.array.max())
    


    # x_fun = [u_fun, m_fun]
    # x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
    
    # i = 0
    # L = dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[i], x_test[i]))
    # grad = dlx.fem.petsc.create_vector(L)
   
    # with grad.localForm() as loc_grad:
    #     loc_grad.set(0)
    
    # dlx.fem.petsc.assemble_vector(grad,L)
    # grad.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    
    # new_func = dlx.fem.Function(Vh[hpx.STATE])

    # ##########################################
    # #works as intended.
    # adj_rhs = dlx.fem.Function(Vh_phi)
    # adj_rhs.interpolate( lambda x: np.log(0.1) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < .5) ) )
    # adj_rhs.x.scatter_forward()
    # adj_rhs = adj_rhs.vector
    # adj_true = pde.generate_state()
    # pde.solveAdj(adj_true,x_true,adj_rhs)    
    # x_true[hpx.ADJOINT] = adj_true

    # eval_grad = pde.evalGradientParameter(x_true)
    # eval_grad_func = hpx.vector2Function(eval_grad,Vh[hpx.PARAMETER])

    # p_fun = hpx.vector2Function(adj_true,Vh[hpx.ADJOINT])

    # mfun = hpx.vector2Function(x_true[hpx.PARAMETER],Vh[hpx.PARAMETER])

    # test_func_handler = H1TikhonvFunctional(3.,4.,5.)
    # test_var_reg = hpx.VariationalRegularization(msh,Vh,test_func_handler,False)

    # test_cost = test_var_reg.cost(x_true[hpx.PARAMETER])

    # test_grad = test_var_reg.grad(x_true[hpx.PARAMETER])
    # test_grad_func = hpx.vector2Function(test_grad,Vh[hpx.PARAMETER])

    # max_iter = 100
    # rel_tol = 1e-6

    # Asolver = petsc4py.PETSc.KSP().create()
    # Asolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
    # Asolver.setType(petsc4py.PETSc.KSP.Type.CG)
    # Asolver.setIterationNumber(max_iter)
    # Asolver.setTolerances(rtol=rel_tol)
    # Asolver.setErrorIfNotConverged(True)
    # Asolver.setInitialGuessNonzero(False)

    # mfun = hpx.vector2Function(x_true[hpx.PARAMETER],Vh[hpx.PARAMETER])

    # test_func_handler = H1TikhonvFunctional(3.,4.,5.)

    # test_obj = dlx.la.create_petsc_vector(Vh[hpx.STATE].dofmap.index_map, Vh[hpx.STATE].dofmap.index_map_bs) 


if __name__ == "__main__":    
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.05, "delta": 1.}
    
    run_inversion(nx, ny, noise_variance, prior_param)

    # def solveFwd(self, state, x): #state is a vector
    #     """ Solve the possibly nonlinear forward problem:
    #     Given :math:`m`, find :math:`u` such that
    #         .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""

    #     mfun = vector2Function(x[PARAMETER],self.Vh[PARAMETER])

    #     # print("first:")
    #     # print(x[PARAMETER].min())
    #     # print(x[PARAMETER].max())

    #     # print(x[PARAMETER],'\n')
    #     # temp_vec = dlx.la.create_petsc_vector(self.Vh[PARAMETER].dofmap.index_map,self.Vh[PARAMETER].dofmap.index_map_bs) 
    #     # temp_vec.x.array[:] = x[PARAMETER]

    #     # temp_m_vec = x[PARAMETER].copy()
    #     # mfun = vector2Function(temp_m_vec,self.Vh[PARAMETER])

    #     #once the function scope ends, the values in the underlying vector also disappear.
    #     #As per: https://fenicsproject.discourse.group/t/manipulating-vector-data-of-fem-function/11056/2
    #     #Need to find a way to work around this.
    #     #Ans: return the function? 


    #     self.n_calls["forward"] += 1
    #     if self.solver is None:
    #         self.solver = self._createLUSolver()

    #     if self.is_fwd_linear:    
    #         u = ufl.TrialFunction(self.Vh[STATE])   
    #         p = ufl.TestFunction(self.Vh[ADJOINT])

    #         res_form = self.varf_handler(u, mfun, p) #all 3 arguments-dl.Function types

    #         # print("second:")
    #         # print(x[PARAMETER].min())
    #         # print(x[PARAMETER].max())
        
    #         A_form = ufl.lhs(res_form) #ufl.form.Form
            
    #         # print("third:")
    #         # print(x[PARAMETER].min())
    #         # print(x[PARAMETER].max()) #garbage

    #         b_form = ufl.rhs(res_form)
            
    #         # print("fourth:")
    #         # print(x[PARAMETER].min()) 
    #         # print(x[PARAMETER].max()) #garbage
        
    #         A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form), bcs=self.bc) #petsc4py.PETSc.Mat
            
    #         # print("fifth:")
    #         # print(x[PARAMETER].min()) #garbage
    #         # print(x[PARAMETER].max()) #garbage
        
    #         A.assemble() #petsc4py.PETSc.Mat
    #         self.solver.setOperators(A)
    #         b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form)) #petsc4py.PETSc.Vec
        
    #         dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])            
    #         b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    #         # dlx.fem.petsc.set_bc(b,self.bc)
    #         self.solver.solve(b,state)
            
    #         return A