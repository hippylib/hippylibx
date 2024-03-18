import dolfinx as dlx
import ufl
import petsc4py
from mpi4py import MPI
import numpy as np
import hippylibX as hpx
import dolfinx as dlx


class VariationalRegularization:
    def __init__(self, Vh : list, functional_handler, isQuadratic=False):
        self.Vh = Vh 
        self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m
        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        

    def cost(self,x : list):
        
        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        
        loc_cost = self.functional_handler(u_fun, m_fun)
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
        dlx.fem.petsc.assemble_vector( tmp_out, dlx.fem.form(ufl.derivative( self.functional_handler(*x_fun), x_fun[i], self.x_test[i]))  )
        tmp_out.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        tmp_out.destroy()


    def setLinearizationPoint(self, x: list, rel_tol=1e-12, max_iter=1000):
        
        hpx.updateFromVector(self.xfun[hpx.STATE], x[hpx.STATE])
        u_fun = self.xfun

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun

        self.x_lin_fn = [u_fun, m_fun]

        L = ufl.derivative(ufl.derivative(self.functional_handler(*self.x_lin_fn),m_fun,self.x_test[hpx.PARAMETER]), m_fun, self.x_test[hpx.PARAMETER])
        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(L))
        self.R.assemble()
        self.Rsolver = petsc4py.PETSc.KSP().create()
        self.Rsolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
        self.Rsolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Rsolver.setIterationNumber(max_iter) 
        self.Rsolver.setTolerances(rtol=rel_tol)
        self.Rsolver.setErrorIfNotConverged(True)
        self.Rsolver.setInitialGuessNonzero(False)
        self.Rsolver.setOperators(self.R)

    def apply_ij(self,i : int,j : int, dir : dlx.la.Vector, out : dlx.la.Vector):
      
        form = self.form(*self.x_lin_fun[hpx.PARAMETER])
        dir_fun = hpx.vector2Function(dir, self.Vh[j])
        action = dlx.fem.form(ufl.derivative( ufl.derivative(form, self.x_lin_fun[i], self.x_test[i]), self.x_lin_fun[j], dir_fun ))
        out.array[:] = 0.
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        dlx.fem.petsc.assemble_vector(tmp_out, action)
        tmp_out.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        tmp_out.destroy()
