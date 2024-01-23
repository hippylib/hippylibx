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

    # print(m_true.min())
    # print(m_true.max())
    
    # #to preserve values in x_true        
    x_true[hpx.STATE] = u_true
    x_true[hpx.PARAMETER] = m_fun_true.vector

    m_true = x_true[hpx.PARAMETER]
    
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

    hpx.random.parRandom(comm,np.sqrt(noise_variance),d.vector) #fix this so it uses SeedSequence    
    # print(rank,":",d.x.array[:])
    
    d.x.scatter_forward()


    # print(rank,":",d.x.array.min(),":",d.x.array.max())

    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = NonGaussianContinuousMisfit(msh, Vh, misfit_form)

    # # PRIOR
    grad = misfit.grad(0,x_true) #different plots in serial and parallel
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


    Vh = Vh_phi #dolfinx.fem.function.FunctionSpace
    # test_obj = Vh._ufl_element.degree() #1
    # print(test_obj)
    qdegree = 2*Vh._ufl_element.degree()
    metadata = {"quadrature_degree" : qdegree}

    # print(type(Vh))

    num_sub_spaces = Vh.num_sub_spaces #0
    # print(num_sub_spaces)

    element = ufl.FiniteElement("Quadrature", Vh.mesh.ufl_cell(), qdegree, quad_scheme="default")
    # element = ufl.VectorElement("Quadrature", Vh.mesh.ufl_cell(), qdegree, dim=num_sub_spaces, quad_scheme="default")

    # Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
    
    Qh = dlx.fem.FunctionSpace(Vh.mesh, element)

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

    prior = hpx.BiLaplacianPrior(Vh_phi,3.,4.,5.)

    model = hpx.Model(pde, prior, misfit)
    noise = prior.init_vector("noise")
    m0 = prior.init_vector(0)


    # noise_func 
    hpx.random.parRandom(comm,1.,noise) #fix this so it uses SeedSequence
    # print(rank,":",m0.getArray())
    prior.sample(noise,m0)
    # print(rank,":",m0.getArray())


    eps, err_grad, _ = hpx.modelVerify(comm, model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))
    # hpx.modelVerify(comm, model,m0, is_quadratic = False, misfit_only=True, verbose = (rank==0))
    
    if(rank == 0):
        print(eps,'\n')
        print(err_grad)
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