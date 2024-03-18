import dolfinx as dlx
import ufl
import petsc4py
from .variables import STATE, PARAMETER, ADJOINT
from ..utils.vector2function import vector2Function, updateFromVector
from mpi4py import MPI
import numpy as np

class VariationalRegularization:
    def __init__(self, Vh, functional_handler, isQuadratic=False):
        self.Vh = Vh #Function space of the parameter.
        self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m
        self.xfun = dlx.fem.Function(self.Vh)
    
    def cost(self,m):    
        updateFromVector(self.xfun, m)
        mfun = self.xfun
        loc_cost = self.functional_handler(mfun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.Vh[STATE].mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )
    
         
    def grad(self, x : dlx.la.Vector, out: dlx.la.Vector) -> None:
        updateFromVector(self.xfun, x)
        mfun = self.xfun
        L = dlx.fem.form(ufl.derivative(self.functional_handler(mfun),mfun,ufl.TestFunction(self.Vh)))
        out.array[:] = 0.
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        tmp_out.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        tmp_out.destroy()

    def setLinearizationPoint(self, m, rel_tol=1e-12, max_iter=1000):
        updateFromVector(self.xfun, m)
        mfun = self.xfun
        L = ufl.derivative(ufl.derivative(self.functional_handler(mfun),mfun), mfun)
        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(L))
        self.R.assemble()
        self.Rsolver = petsc4py.PETSc.KSP().create()
        self.Rsolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
        self.Rsolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Rsolver.setIterationNumber(max_iter) #these values should be supplied as arguments.
        self.Rsolver.setTolerances(rtol=rel_tol)
        self.Rsolver.setErrorIfNotConverged(True)
        self.Rsolver.setInitialGuessNonzero(False)
        self.Rsolver.setOperators(self.R)
