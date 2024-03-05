import dolfinx as dlx
import ufl
import petsc4py
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function, updateFromVector
from mpi4py import MPI
import numpy as np

class VariationalRegularization:
    def __init__(self, mesh, Vh, functional_handler, isQuadratic=False):
        self.Vh = Vh #Function space of the parameter.
        self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m
        self.mesh = mesh #to allreduce over the entire mesh in the cost function

        self.xfun = dlx.fem.Function(self.Vh)

    # def cost(self, m):
    # 1. Cast the petsc4py vector m to a dlx.Function mfun
    # 2. Compute the cost by calling assemble on self.functional_handler(mfun)
    # 3. Return the value of the cost (Make sure to call a AllReduce for parallel computations
    
    def cost(self,m):
        # mfun = vector2Function(m,self.Vh)
    
        updateFromVector(self.xfun, m)
        mfun = self.xfun

        loc_cost = self.functional_handler(mfun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )
    

    # def grad(self, m, out):
    #  1. Cast the petsc4py vector m to a dlx.Function mfun
    #  2. call symbolic differentation of self.functional_handler(mfun) wrt mfun
    #  3. call assemble, update ghosts, and store the result in out
         
    def grad(self, x : dlx.la.Vector, out: dlx.la.Vector) -> None:
        # mfun = vector2Function(x,self.Vh)

        updateFromVector(self.xfun, x)
        mfun = self.xfun


        L = dlx.fem.form(ufl.derivative(self.functional_handler(mfun),mfun,ufl.TestFunction(self.Vh)))
        tmp = dlx.fem.petsc.assemble_vector(L)
        tmp.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        dlx.la.create_petsc_vector_wrap(out).scale(0.)
        dlx.la.create_petsc_vector_wrap(out).axpy(1., tmp)

    # def setLinearizationPoint(self, m):
    #   1. Cast the petsc4py vector m to a dlx.Function mfun
    #   2. call symbolic differentiation (twice) to get the second variation of self.functional_handler(mfun) wrt mfun
    #   3. assemble the Hessian operator (it's a sparse matrix!) in the attribute self.R
    #   4. set up a linearsolver self.Rsolver that uses CG as Krylov method, gamg as preconditioner and self.R as operator

    def setLinearizationPoint(self, m, rel_tol=1e-12, max_iter=1000):
        # mfun = vector2Function(m,self.Vh)

        updateFromVector(self.xfun, m)
        mfun = self.xfun

        L = ufl.derivative(ufl.derivative(self.functional_handler(mfun),mfun), mfun)
        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(L))
        self.R.assemble()
        
        #for Rsolver:
        #have to call _BilaplacianRsolver(self.Asolver,self.M)
        #so have to construct self.Asolver, self.M first
        
        self.Rsolver = petsc4py.PETSc.KSP().create()
        self.Rsolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
        self.Rsolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Rsolver.setIterationNumber(max_iter) #these values should be supplied as arguments.
        self.Rsolver.setTolerances(rtol=rel_tol)
        self.Rsolver.setErrorIfNotConverged(True)
        self.Rsolver.setInitialGuessNonzero(False)
        self.Rsolver.setOperators(self.R)
