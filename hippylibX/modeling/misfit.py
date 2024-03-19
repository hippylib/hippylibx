import hippylibX as hpx
import dolfinx as dlx
import ufl
from mpi4py import MPI
import petsc4py
import numpy as np

class NonGaussianContinuousMisfit(object):
    def __init__(self,Vh : list, form, bc=[]):
        self.Vh = Vh
        self.form = form
        self.bc = bc

        self.x_lin_fun = None
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        self.gauss_newton_approx = False

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    def cost(self,x : list) -> float:

        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        loc_cost = self.form(u_fun,m_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.Vh[hpx.STATE].mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )

    def grad(self, i : int, x : list, out: dlx.la.Vector) -> dlx.la.Vector:
    
        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        x_fun = [u_fun, m_fun]
        
        out.array[:] = 0.
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        dlx.fem.petsc.assemble_vector( tmp_out, dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i]))  )
        tmp_out.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        dlx.fem.petsc.set_bc(tmp_out,self.bc)
        tmp_out.destroy()


    def setLinearizationPoint(self,x : list, gauss_newton_approx=False):
        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]
        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx 
        

    def apply_ij(self,i : int,j : int, dir : dlx.la.Vector, out : dlx.la.Vector):

        form = self.form(*self.x_lin_fun)
        tmp_dir = dlx.la.create_petsc_vector_wrap(dir)
        if(j == hpx.STATE):
            dlx.fem.petsc.set_bc(tmp_dir,self.bc)
        tmp_dir.destroy()

        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = dlx.fem.form(ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun ))        
        out.array[:] = 0.
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        dlx.fem.petsc.assemble_vector(tmp_out, action)
        tmp_out.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        if(i == hpx.STATE):
            dlx.fem.petsc.set_bc(tmp_out,self.bc)
        tmp_out.destroy()
