import hippylibX as hpx
import dolfinx as dlx
import ufl
from mpi4py import MPI
import petsc4py

class NonGaussianContinuousMisfit(object):
    def __init__(self, mesh, Vh, form):
        #mesh needed for comm.allreduce in cost function
        self.mesh = mesh
        self.Vh = Vh
        self.form = form

        self.x_lin_fun = None
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        self.gauss_newton_approx = False

    def cost(self,x):
        u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        u_fun.x.scatter_forward()
        m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])
        m_fun.x.scatter_forward()  
        loc_cost = self.form(u_fun,m_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )

    def grad(self, i, x):

        u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])

        #This needs to be parallelized:
        # dl.assemble(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i]), tensor=out )
        
        # u_fun.x.scatter_forward()
        # m_fun.x.scatter_forward()

        # print(self.mesh.comm.rank,":",m_fun.x.array.min(),":",m_fun.x.array.max())

        # u_fun_test = ufl.TestFunction(self.Vh[hpx.STATE])
        # m_fun_test =  ufl.TestFunction(self.Vh[hpx.PARAMETER])
        # u_fun_test.x.scatter_forward()
        # m_fun_test.x.scatter_forward()

        x_fun = [u_fun, m_fun]
        x_test = [ufl.TestFunction(self.Vh[hpx.STATE]), ufl.TestFunction(self.Vh[hpx.PARAMETER])]

        L = dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], x_test[i]))
    
        out =  dlx.fem.assemble_vector(L)
        dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        return out
    
        #need to return a dlx.la.Vector object instead


        #substitute for dl.assemble:

        #M-1 ###################################
        # out = dlx.fem.Function(self.Vh[i])
        # dlx.fem.petsc.assemble_vector(out.vector,L)
        # out.x.scatter_reverse(dlx.la.ScatterMode.add)
        # out.x.scatter_forward()
        # return out.vector
        # ##################################


        #M-2 ###################################
        out = dlx.fem.petsc.create_vector(L)

        with out.localForm() as loc_grad:
            loc_grad.set(0)
        
        dlx.fem.petsc.assemble_vector(out,L)
        out.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        # out.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        
        return out
    #    ##################################
 
        # x_state_fun,x_par_fun = hpx.vector2Function(x[hpx.STATE],self.Vh[hpx.STATE]), hpx.vector2Function(x[hpx.PARAMETER],self.Vh[hpx.PARAMETER]) 
        # x_fun = [x_state_fun,x_par_fun] 
        # loc_grad = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i])) ) #<class 'PETSc.Vec'>
        # loc_grad.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        # return loc_grad
    
    # def grad(self, i, x, out):

    #     u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
    #     m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])

    #     #This needs to be parallelized:
    #     # dl.assemble(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i]), tensor=out )
        
    #     # u_fun.x.scatter_forward()
    #     # m_fun.x.scatter_forward()

    #     # print(self.mesh.comm.rank,":",m_fun.x.array.min(),":",m_fun.x.array.max())

    #     # u_fun_test = ufl.TestFunction(self.Vh[hpx.STATE])
    #     # m_fun_test =  ufl.TestFunction(self.Vh[hpx.PARAMETER])
    #     # u_fun_test.x.scatter_forward()
    #     # m_fun_test.x.scatter_forward()

    #     x_fun = [u_fun, m_fun]
    #     x_test = [ufl.TestFunction(self.Vh[hpx.STATE]), ufl.TestFunction(self.Vh[hpx.PARAMETER])]

    #     L = dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], x_test[i]))
    
    #     out =  dlx.fem.assemble_vector(L)
    #     dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    #     return out
    
    #     #need to return a dlx.la.Vector object instead


    #     #substitute for dl.assemble:

    #     #M-1 ###################################
    #     # out = dlx.fem.Function(self.Vh[i])
    #     # dlx.fem.petsc.assemble_vector(out.vector,L)
    #     # out.x.scatter_reverse(dlx.la.ScatterMode.add)
    #     # out.x.scatter_forward()
    #     # return out.vector
    #     # ##################################


    #     #M-2 ###################################
    #     out = dlx.fem.petsc.create_vector(L)

    #     with out.localForm() as loc_grad:
    #         loc_grad.set(0)
        
    #     dlx.fem.petsc.assemble_vector(out,L)
    #     out.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    #     # out.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        
    #     return out
    # #    ##################################
 
    #     # x_state_fun,x_par_fun = hpx.vector2Function(x[hpx.STATE],self.Vh[hpx.STATE]), hpx.vector2Function(x[hpx.PARAMETER],self.Vh[hpx.PARAMETER]) 
    #     # x_fun = [x_state_fun,x_par_fun] 
    #     # loc_grad = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i])) ) #<class 'PETSc.Vec'>
    #     # loc_grad.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
    #     # return loc_grad


    
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx 
        

    #same issue as grad function, what is "type" of out that has to be passed as argument?
    def apply_ij(self,i,j, dir):
        form = self.form(*self.x_lin_fun)
        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = dlx.fem.form(ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun ))
        
        apply = dlx.fem.petsc.create_vector(action)
        with apply.localForm() as loc_apply:
            loc_apply.set(0)
        dlx.fem.petsc.assemble_vector(apply,action)

        apply.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        return apply

        
