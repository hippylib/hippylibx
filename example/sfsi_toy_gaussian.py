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
    
    
    # #LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    expr = u_fun * ufl.exp(m_fun)
    hpx.projection(expr,d)
    hpx.random.parRandom(comm,np.sqrt(noise_variance),d)
    d.x.scatter_forward()

    misfit_form = PACTMisfitForm(d, noise_variance)
    misfit = NonGaussianContinuousMisfit(msh, Vh, misfit_form)  

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
    ans = misfit.grad(0,x_true)

    print(rank,":",min(ans.array))
    print(rank,":",max(ans.array))

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