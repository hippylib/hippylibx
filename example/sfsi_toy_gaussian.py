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

    print(rank,":",misfit.cost(x_true))


    # loc_cost = misfit_form(u_fun,m_fun)     

    # print(rank,":",loc_cost)

    # loc_cost = misfit_form(u_fun,m_fun) #ufl.form.Form

    # glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))

    # glb_cost_proc.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

    # print(rank,":",glb_cost_proc)
    # glb_cost = msh.comm.allreduce(glb_cost_proc, op=MPI.SUM )
    
    # print(rank,":",glb_cost)


    # glb_cost.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

    # V_test = dlx.fem.FunctionSpace(msh,("CG",1))
    # expr = dlx.fem.Expression( dlx.fem.form(loc_cost) ,  V_test.element.interpolation_points())
    

    # glb_cost = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))

    # min_val = msh.comm.allreduce(min(u_fun.x.array), op=MPI.MIN)
    # max_val = msh.comm.allreduce(max(u_fun.x.array), op=MPI.MAX)
    # if msh.comm.rank == 0:
    #     print(min_val, max_val)

    # 
    # print(type(glb_cost))
    # msh.comm.allreduce(glb_cost,op=MPI.SUM)

    # print(rank,":",glb_cost)
        
    # print(u_fun.vector[:].min())
    # print(u_fun.vector[:].max(),'\n')

    # print(m_fun.vector[:].min())
    # print(m_fun.vector[:].max(),'\n')

    # print(x_true[hpx.STATE].min())
    # print(x_true[hpx.STATE].max(),'\n')

    # print(x_true[hpx.PARAMETER].min())
    # print(x_true[hpx.PARAMETER].max(),'\n')
    
    #as expected
    # print(m_fun_true.vector.min())
    # print(m_fun_true.vector.max())
        
    # m_true_3 = m_fun_true.vector
    # print(m_true.min())

    # print(m_true.max())
    # temp_vec = dlx.la.create_petsc_vector(Vh[hpx.PARAMETER].dofmap.index_map,Vh[hpx.PARAMETER].dofmap.index_map_bs) 
    # temp_vec = m_true.copy()
    # temp_vec[0] = -5

    # print(m_true_org.min())
    # print(m_true_org.max(),'\n')

    # print(m_true_2.min())
    # print(m_true_2.max(),'\n')

    # print(m_true_3.min())
    # print(m_true_3.max(),'\n')

    # print(m_true_4.min())
    # print(m_true_4.max(),'\n')

    # print(m_true_2_vec.min())
    # print(m_true_2_vec.max(),'\n')

    # print(m_true_3.min())
    # print(m_true_3.max(),'\n')

    # print(temp_vec2.min())
    # print(temp_vec2.max(),'\n')


    # print(x_true[hpx.PARAMETER].min())
    # print(x_true[hpx.PARAMETER].max())
    
    # #done to ensure values restored in m_true after solveFwd
    # x_true[hpx.PARAMETER] = m_fun_true.vector

    # #LIKELIHOOD
    
    # u_fun_true = hpx.vector2Function(u_true, Vh[hpx.STATE]) 
    # d = dlx.fem.Function(Vh[hpx.STATE])
    # expr = u_fun_true * ufl.exp(m_fun_true)
    # hpx.projection(expr,d)
    # hpx.random.parRandom(comm,noise_variance,d)
    
    # misfit_form = PACTMisfitForm(d, noise_variance)
    # misfit = NonGaussianContinuousMisfit(msh, Vh, misfit_form)
    
    # # loc_cost = misfit.cost(x_true)
    # loc_cost = misfit_form(u_fun_true,m_fun_true)
    # glb_cost = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))

    # print(rank,":",glb_cost)

    # print(x_true[hpx.STATE][:].min())
    # print(x_true[hpx.STATE][:].max(),'\n')
        
    # print(x_true[hpx.PARAMETER][:].min())
    # print(x_true[hpx.PARAMETER][:].max())
    

    # print(rank,":",cost)


    # v = petsc4py.PETSc.Viewer()
    # v(d.vector)
    # print(rank,":",d.x.array[:10])

    # noise_variance = dlx.Constant(msh,petsc4py.PETSc.ScalarType(noise_variance))

        
    # cost = misfit.cost(x_true)
    # print(rank,":",cost)

    # u_fun_true = hpx.vector2Function(x_true[hpx.STATE], Vh[hpx.STATE])
    #m_fun_true already made


    # print(m_fun_true.x.array[:].min())
    # print(m_fun_true.x.array[:].max())

    # loc_cost = misfit_form(u_fun,m_fun) #ufl.form.Form
    # print(rank,":",loc_cost)

    # glb_cost = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
    # print(rank,"cost 1:",glb_cost)

    # glb_cost = dlx.fem.assemble.assemble_scalar(dlx.fem.form(loc_cost))
    
    # print(rank,"cost 2:",glb_cost)


    # msh.comm.allreduce(loc_cost,op=MPI.SUM)
    # glb_cost = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))


    # print(type(loc_cost))  


    # with dlx.io.XDMFFile(msh.comm, "attempt_project_np{0:d}_X.xdmf".format(nproc),"w") as file: #works!!
    #     file.write_mesh(msh)
    #     file.write_function(d)


if __name__ == "__main__":
    
    nx = 64
    ny = 64
    noise_variance = 1e-6
    prior_param = {"gamma": 0.05, "delta": 1.}
    
    run_inversion(nx, ny, noise_variance, prior_param)