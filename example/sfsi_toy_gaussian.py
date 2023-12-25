import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import petsc4py

import sys
import os

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
    # x_true = [u_true, m_true_org, None]     #list of petsc vectors

    # temp_vec = m_true.copy()
    # m_true_2 = dlx.la.create_petsc_vector(Vh[hpx.PARAMETER].dofmap.index_map,Vh[hpx.PARAMETER].dofmap.index_map_bs) 
    # m_true_2.axpy(1,m_true_org)
    
    # m_true_3 = m_true_2.copy()

    #only original vector gets messed up if you do 1 or 2 levels.

    #what if use vetor associated with a duplicate function - still doesn't work
    # m_true_2_func = dlx.fem.Function(Vh_m)
    # m_true_2_vec = m_true_2_func.vector
    # m_true_2_vec.axpy(1,m_true_org)

    # m_true_2 = dlx.fem.Function(Vh_m)
    # m_true_2.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )) # <class 'dolfinx.fem.function.Function'>
    # m_true_2.x.scatter_forward()  
    # m_fun_true = dlx.fem.Function(Vh_m)
    # m_fun_true.x.array[:] = m_true.x.array[:]

    # m_true_2 = m_true_2.vector #<class 'petsc4py.PETSc.Vec'>
    
    # m_true_4 = dlx.fem.Function(Vh_m)
    # m_true_4.interpolate(lambda x: np.log(0.1) + 4.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1 ) )) # <class 'dolfinx.fem.function.Function'>
    # m_true_4.x.scatter_forward()  
    # m_true_4 = m_true_4.vector 

    # print(m_true_4.min())
    # print(m_true_4.max())
      
    #created 2 entriely different vectors, pass any one to solveFwd
    #then why are both turned to garbage??
    
    # print(m_true_org)
    # print(m_true_2)
    
    # print("m_true_org",m_true_org,'\n')
    # print("m_true2",m_true_2,'\n')
    # print("m_true3",m_true_3,'\n')
    
    
    # print(temp_vec)
    # print(temp_vec2)
    
    # print(temp_vec.min())
    # print(temp_vec.max(),'\n')
    
    # print(m_true.min())
    # print(m_true.max(),'\n')

    x_true = [u_true, m_true, None]     #list of petsc vectors

    # x_true2 = [u_true,m_true_2,None]
    
    #M-1
    # m_true_fun = pde.solveFwd(u_true, x_true) #petsc4py.PETSc.Vec
    # print(m_true_fun.vector[:].min())
    # print(m_true_fun.vector[:].max())
    
    #M-2
    # print(m_true_org.min())
    # print(m_true_org.max(),'\n')

    # print(m_true_2.min())
    # print(m_true_2.max(),'\n')
    
    pde.solveFwd(u_true, x_true) #petsc4py.PETSc.Vec

    #to preserve values in x_true        
    x_true[hpx.STATE] = u_true


    x_true[hpx.PARAMETER] = m_fun_true.vector

    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    u_fun.x.scatter_forward()
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    m_fun.x.scatter_forward()
    
    # print(rank,":",min(u_fun.x.array),":",max(u_fun.x.array))


    # LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)
    hpx.random.parRandom(comm,np.sqrt(noise_variance),d)
    d.x.scatter_forward()

    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = NonGaussianContinuousMisfit(msh, Vh, misfit_form)  

    #implement the gradient evaluation function before implementing the regularization

    # PRIOR

    #works as expected
    # print(rank,":",misfit.cost(x_true))

    # u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    # m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    # u_fun.x.scatter_forward()
    # m_fun.x.scatter_forward()

    # x_fun = [u_fun, m_fun]
    # x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]

    # i = 0
    # # ans = dlx.fem.petsc.assemble_vector(dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[i], x_test[i])) )
    # L = dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[i], x_test[i]))


    # ans = dlx.fem.petsc.create_vector(L)
    # with ans.localForm() as loc_ans:
    #     loc_ans.set(0)
    # dlx.fem.petsc.assemble_vector(ans,L)
    # ans.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    # ans.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
 
    # msh.comm.allreduce()
    # ans.scatter_reverse()

    # print(x_true[hpx.STATE].min())
    # print(x_true[hpx.STATE].max())

    # print(x_true[hpx.PARAMETER].min())
    # print(x_true[hpx.PARAMETER].max())
    
    # ans = misfit.grad(0,x_true)

    u_fun = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    m_fun = hpx.vector2Function(x_true[hpx.PARAMETER], Vh[hpx.PARAMETER])
    
    # print(rank,":",len(u_fun.x.array))

    # print(rank,":",u_fun.vector.min())
    # print(rank,":",u_fun.vector.max())

    # print(rank,":",m_fun.vector.min())
    # print(rank,":",m_fun.vector.max())
    
    u_fun.x.scatter_forward()
    m_fun.x.scatter_forward()

    x_fun = [u_fun, m_fun]
    x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
    i = 0

    L = dlx.fem.form(ufl.derivative( misfit_form(*x_fun), x_fun[i], x_test[i]))
    grad = dlx.fem.petsc.create_vector(L)
    
    # print(rank,":",len(np.array(u_fun.x.array)),":",len(np.array(grad)))

    with grad.localForm() as loc_grad:
        loc_grad.set(0)
    
    dlx.fem.petsc.assemble_vector(grad,L)
    grad.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)

    #try to allreduce values in grad for each process to all others?
    # create new vector of needed size (3287)
    new_func = dlx.fem.Function(Vh[hpx.STATE])

    
    # work_sizes = [818,816,831,822]
    
    #But I dont know in what order/ displacement each process has the values in
    # grad. It may not even be in contiguous blocks. - Cannot use Allgatherv :-(

    # msh.comm.Allgatherv(grad,new_func.vector)


    # print(rank,":",len(np.array(u_fun.x.array)),":",len(np.array(grad)))
    # print(rank, ":",grad.array.min(),":",grad.array.max() ) 
    # print(rank,":",misfit.cost(x_true))        

    #Question: Are all processes suppsed to have same vector values returned from grad?
    #No, I guess, because in normal hippylib, they dont have the same values either.

    #move on for now?
    #eval Gradient function?? -> p from adjoint, then use eval Gradient?
    #to solveAdj, need adj_rhs. create dummy??

    ##########################################
    # adj_rhs = dlx.fem.Function(Vh_phi)
    # adj_rhs.interpolate( lambda x: np.log(0.1) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < .5) ) )
    # adj_rhs.x.scatter_forward()
    # adj_rhs = adj_rhs.vector



    # need to do something similar to assemble_vector as in the solveFwd method.
    # ^ alternative to above
    ##########################################
    
    ##########################################
    ###doesn't work either
    adj_rhs = pde.generate_state()
    adj_rhs_func = hpx.vector2Function(adj_rhs,Vh[hpx.ADJOINT])
    adj_rhs_func.interpolate( lambda x: np.log(0.1) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < .5) ) )
    adj_rhs_func.x.scatter_forward()
    adj_rhs = adj_rhs_func.vector

    # print(rank,":",adj_rhs.array.min(),":",adj_rhs.array.max())
    
    ##########################################
    
    # adj_rhs.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    # adj_rhs_expr = dl.Expression("std::log(0.1) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < .5) )", degree=1)
    # adj_rhs = dl.interpolate(adj_rhs_expr, Vh_m).vector()

    adj_true = pde.generate_state()
    
    # # print(type(adj_rhs))
    # #need to get this to work in parallel - not working!
    
    pde.solveAdj(adj_true,x_true,adj_rhs)    

    # print(rank,":",adj_rhs.array.min(),":",adj_rhs.array.max())
    # print(rank,":",adj_true.array.min(),":",adj_true.array.max())

    adj_true.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    adj_true.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
    
    x_true[hpx.ADJOINT] = adj_true

    adj_true_func = hpx.vector2Function(adj_true,Vh[hpx.ADJOINT])  

    # min_val = msh.comm.allreduce(min(adj_true_func.x.array), op=MPI.MIN)
    # max_val = msh.comm.allreduce(max(adj_true_func.x.array), op=MPI.MAX)

    # if msh.comm.rank == 0:
    #     print(min_val, max_val)    

    # print(rank,":",x_true[hpx.STATE].array.min(),":",x_true[hpx.STATE].array.max())
    # print(rank,":",x_true[hpx.PARAMETER].array.min(),":",x_true[hpx.PARAMETER].array.max())    

    # adj_fun_true = hpx.vector2Function(x_true[hpx.ADJOINT],Vh[hpx.ADJOINT])
    # adj_fun_true.x.scatter_forward()


    #works as intended in both serial and parallel
    with dlx.io.XDMFFile(msh.comm, "attempt_adjoint_grad_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
        file.write_mesh(msh)
        file.write_function(adj_true_func)

    # print(rank,":",min(adj_fun_true.x.array),":",max(adj_fun_true.x.array))

    # print(rank,":",min(adj_true.array),":",max(adj_true.array))
    
    # print(rank,":",min(ans.array))
    # print(rank,":",max(ans.array))

    # ans_func = hpx.vector2Function(ans,Vh[hpx.STATE])

    # with dlx.io.XDMFFile(msh.comm, "attempt_misfit_grad_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(ans_func)

    # print(rank,":",min(ans_func.x.array))
    # print(rank,":",max(ans_func.x.array))

    # print(rank,":",min(ans.array))
    # print(rank,":",max(ans.array))

    # print(dlx.fem.form(L).function_spaces[0].dofmap.index_map)

    # ans = dlx.fem.petsc.assemble_vector(dlx.fem.form(L))    
    # ans.scatter_reverse
    # ans.assemble()

    # ans.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    # ans.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    
    # ans.assemble()
    
    # ans = dlx.fem.petsc.assemble_vector(dlx.fem.form(L) )
    # ans = dlx.fem.petsc.assemble_vector( L )

    # with ans.localForm() as ans_local:
    #     print(ans_local.array)


    # ans = dlx.fem.assemble_vector(dlx.fem.form(L)) #dolfinx.la.Vector
    # ans.scatter_reverse

    # print(rank,":",len(ans.array))

    # print(rank,":",ans.get_local())
    
    # test_obj = ans.get_local()
    # print("hello")
    # print(rank,":",ans.get_local())
    # print(ans.array[:].min())
    # print(ans.array[:].max())

    # misfit_grad_func = dlx.fem.Function(Vh[hpx.STATE])

    # print(rank,":",len(ans.array),":",len(misfit_grad_func.x.array[:]))

    # misfit_grad_func.x.array[:] = ans.array

    # with ans.localForm() as ans_local:
    #     print(rank,":",ans_local.array.max())

    # print(type(ans))
    # ans.assemble()

    # misfit_grad_func = hpx.vector2Function(ans,Vh[hpx.STATE])

    # misfit_grad_func.x.scatter_forward()

    # min_val = msh.comm.allreduce(min(misfit_grad_func.x.array), op=MPI.MIN)
    # max_val = msh.comm.allreduce(max(misfit_grad_func.x.array), op=MPI.MAX)


    # min_val = msh.comm.allreduce(min(m_fun.x.array), op=MPI.MIN)
    # max_val = msh.comm.allreduce(max(m_fun.x.array), op=MPI.MAX)

    # if msh.comm.rank == 0:
    #     print(min_val, max_val)    


    # with dlx.io.XDMFFile(msh.comm, "attempt_misfit_grad_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(misfit_grad_func)


    # ans  =  ufl.derivative( misfit_form(*x_fun), x_fun[i], x_test[i])
    
    # print('hello')


 
    # with dlx.io.XDMFFile(msh.comm, "attempt_project_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(d)


if __name__ == "__main__":
    
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.05, "delta": 1.}
    
    run_inversion(nx, ny, noise_variance, prior_param)