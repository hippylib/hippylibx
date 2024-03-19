import dolfinx as dlx
import ufl
import petsc4py
from mpi4py import MPI
import numpy as np
import hippylibX as hpx
import dolfinx as dlx


# an example of functional_handler can be:
# class H1TikhonvFunctional:
#     def __init__(self, gamma, delta, m0):	
#         self.gamma = gamma #These are dlx Constant, Expression, or Function
#         self.delta = delta
#         self.m0 = m0

#     def __call__(self, m): #Here m is a dlx Function
#         return ufl.inner(self.gamma * ufl.grad(m), ufl.grad(m) ) *ufl.dx + \
#                ufl.inner(self.delta * m, m)*ufl.dx


class VariationalRegularization:
    def __init__(self, Vh : list, functional_handler, isQuadratic=False):
        # self.Vh = Vh #Function space of the parameter.
        # self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        # self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m


        self.Vh = Vh 
        self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m
        self.xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]
        self.x_test = [ufl.TestFunction(Vh[hpx.STATE]), ufl.TestFunction(Vh[hpx.PARAMETER])]
        

    def cost(self,x : list):
        
        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. Compute the cost by calling assemble on self.functional_handler(mfun)
        # 3. Return the value of the cost (Make sure to call a AllReduce for parallel computations

        hpx.updateFromVector(self.xfun[hpx.STATE],x[hpx.STATE])
        u_fun = self.xfun[hpx.STATE]

        hpx.updateFromVector(self.xfun[hpx.PARAMETER], x[hpx.PARAMETER])
        m_fun = self.xfun[hpx.PARAMETER]
        
        loc_cost = self.functional_handler(u_fun, m_fun)
        glb_cost_proc = dlx.fem.assemble_scalar(dlx.fem.form(loc_cost))
        return self.Vh[hpx.STATE].mesh.comm.allreduce(glb_cost_proc, op=MPI.SUM )


    def grad(self, i : int, x : list, out: dlx.la.Vector) -> dlx.la.Vector:

        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. call symbolic differentation of self.functional_handler(mfun) wrt mfum
        # 3. call assemble, update ghosts, and store the result in out

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

        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. call symbolic differentiation (twice) to get the second variation of self.functional_handler(mfun) wrt mfun
        # 3. assemble the Hessian operator (it's a sparse matrix!) in the attribute self.R
        # 4. set up a linearsolver self.Rsolver that uses CG as Krylov method, gamg as preconditioner and self.R as operator

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