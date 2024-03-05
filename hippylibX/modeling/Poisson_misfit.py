import hippylibX as hpx
import dolfinx as dlx
import ufl
from mpi4py import MPI
import petsc4py

class PoissonMisfit(object):
    def __init__(self, mesh : dlx.mesh.Mesh, Vh : list, form):
        #mesh needed for comm.allreduce in cost function
        self.mesh = mesh
        self.Vh = Vh
        self.form = form

        self.x_lin_fun = None
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        self.gauss_newton_approx = False

        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]

    def cost(self,x : list) -> float:
        # u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        # m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])

        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        loc_cost = self.form(u_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )

    def grad(self, i : int, x : list, out: dlx.la.Vector) -> dlx.la.Vector:

        # u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        # m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])
    
        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]

        x_fun = [u_fun]
        
        L = dlx.fem.form(ufl.derivative( self.form(*x_fun), x_fun[i], self.x_test[i]))
        tmp =  dlx.fem.petsc.assemble_vector(L)
        tmp.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)

        # dlx.la.create_petsc_vector_wrap(out).scale(0.)
        # dlx.la.create_petsc_vector_wrap(out).axpy(1., tmp)
        
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)
        temp_petsc_vec_out.scale(0.)
        temp_petsc_vec_out.axpy(1., tmp)
        tmp.destroy()
        temp_petsc_vec_out.destroy()

    def setLinearizationPoint(self,x : list, gauss_newton_approx=False):
        
        # u_fun = hpx.vector2Function(x[hpx.STATE], self.Vh[hpx.STATE])
        # m_fun = hpx.vector2Function(x[hpx.PARAMETER], self.Vh[hpx.PARAMETER])

        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER],x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        
        self.x_lin_fun = [u_fun, m_fun]
        self.gauss_newton_approx = gauss_newton_approx 
        

    def apply_ij(self,i : int,j : int, dir : dlx.la.Vector, out : dlx.la.Vector):
        
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)

        # dlx.la.create_petsc_vector_wrap(out).scale(0.)
        temp_petsc_vec_out.scale(0.)

        form = self.form(*self.x_lin_fun)
        
        dir_fun = hpx.vector2Function(dir, self.Vh[j])

        # hpx.updateFromVector(self.xfun[j],dir)
        # dir_fun = self.xfun[j]

        action = dlx.fem.form(ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun ))
        tmp = dlx.fem.petsc.assemble_vector(action)
        tmp.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        # dlx.la.create_petsc_vector_wrap(out).axpy(1., tmp)
        temp_petsc_vec_out.axpy(1., tmp)
        temp_petsc_vec_out.destroy()
        tmp.destroy()
        


