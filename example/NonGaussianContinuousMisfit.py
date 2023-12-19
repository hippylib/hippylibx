import hippylibX as hpx
import dolfinx as dlx
import ufl
from mpi4py import MPI
import petsc4py

class NonGaussianContinuousMisfit(object):
    def __init__(self, mesh, Vh, form):
        self.mesh = mesh
        self.Vh = Vh
        self.form = form

        self.x_lin_fun = None
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        self.gauss_newton_approx = False

    def cost(self,x):
        loc_cost = dlx.fem.assemble_scalar(dlx.fem.form(self.form(hpx.vector2Function(x[hpx.STATE],self.Vh[hpx.STATE]), hpx.vector2Function(x[hpx.PARAMETER],self.Vh[hpx.PARAMETER]))))
        return self.mesh.comm.allreduce(loc_cost,op=MPI.SUM)

    def grad(self, i, x):
        x_state_fun,x_par_fun = hpx.vector2Function(x[hpx.STATE],self.Vh[hpx.STATE]), hpx.vector2Function(x[hpx.PARAMETER],self.Vh[hpx.PARAMETER]) 
        x_fun = [x_state_fun,x_par_fun] 
        loc_grad = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i])) ) #<class 'PETSc.Vec'>
        loc_grad.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        return loc_grad
    
      
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx 
        

    #same issue as grad function, what is "type" of out that has to be passed as argument?
    def apply_ij(self,i,j, dir):
        form = self.form(*self.x_lin_fun)
        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun )
        loc_action = dlx.fem.petsc.assemble.assemble_vector(dlx.fem.form(action) ) #<class 'PETSc.Vec'>
        loc_action.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        return loc_action
