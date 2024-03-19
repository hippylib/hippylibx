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
    def __init__(self, Vh : dlx.FunctionSpace, functional_handler, isQuadratic=False):
        # self.Vh = Vh #Function space of the parameter.
        # self.functional_handler = functional_handler #a function or a functor that takes as input m (as dl.Function) and evaluate the regularization functional
        # self.isQuadratic = isQuadratic #Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian dependes on m


        self.Vh = Vh 
        self.functional_handler = functional_handler #a function or a functor that takes as input m (as dlx.Function) and evaluates the regularization functional
        self.isQuadratic = isQuadratic # Whether the functional is a quadratic form (i.e. the Hessian is constant) or not (the Hessian depends on m
        self.mfun = dlx.fem.Function(Vh)
        self.mtest = ufl.TestFunction(Vh)
        self.mtrial = ufl.TrialFunction(Vh)

        self.petsc_option_M = {"ksp_type": "cg", "pc_type": "jacobi"} # see if you can set rtol atol maxiter
        self.petsc_option_R = {"ksp_type": "cg", "pc_type": "hypre"}

        self.R = None
        self.Rsolver= None
        
        if(self.isQuadratic == True):
            tmp = dlx.fem.Function(Vh).x
            self.setLinearizationPoint(tmp)

        self.M = dlx.fem.petsc.assemble( dlx.form( ufl.inner(self.mtrial, self.mtest)*ufl.dx ) )
        self.M.assemble()
        self.Msolver = ......

    def __del__(self):
        self.Rsolver.destroy()
        self.R.destroy()
        self.Msolver.destroy()
        self.M.destroy()

    def cost(self,m : dlx.la.Vector):
        
        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. Compute the cost by calling assemble on self.functional_handler(mfun)
        # 3. Return the value of the cost (Make sure to call a AllReduce for parallel computations

        hpx.updateFromVector(self.mfun,m)
        cost_functional = self.functional_handler(mfun)
        local_cost = dlx.fem.assemble_scalar(dlx.fem.form(cost_functional))
        return self.Vh.mesh.comm.allreduce(local_cost, op=MPI.SUM )


    def grad(self, m: dlx.la.Vector, out: dlx.la.Vector) -> dlx.la.Vector:

        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. call symbolic differentation of self.functional_handler(mfun) wrt mfum
        # 3. call assemble, update ghosts, and store the result in out

        hpx.updateFromVector(self.mfun,m)

        out.array[:] = 0.
        tmp_out = dlx.la.create_petsc_vector_wrap(out)
        dlx.fem.petsc.assemble_vector( tmp_out, dlx.fem.form(ufl.derivative( self.functional_handler(mfun), mfun[i], self.mtest))  )
        tmp_out.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        tmp_out.destroy()


    def setLinearizationPoint(self, m: dlx.la.Vector, rel_tol=1e-12, max_iter=1000): #remove rel_tol, max_iter from this signature

        # 1. Cast the petsc4py vector m to a dlx.Function mfun
        # 2. call symbolic differentiation (twice) to get the second variation of self.functional_handler(mfun) wrt mfun
        # 3. assemble the Hessian operator (it's a sparse matrix!) in the attribute self.R
        # 4. set up a linearsolver self.Rsolver that uses CG as Krylov method, gamg as preconditioner and self.R as operator

        if (self.isQuadratic == True) and (self.R is not None):
            return

        hpx.updateFromVector(self.mfun, m)

        L = ufl.derivative(ufl.derivative(self.functional_handler(self.mfun), self.mfun,self.mtest), self.mfun, self.mtrial)
        self.R = dlx.fem.petsc.assemble_matrix(dlx.fem.form(L))
        self.R.assemble()
        self.Rsolver = petsc4py.PETSc.KSP().create()
        #UPDATE USING A PETSC OPTION.
        self.Rsolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
        self.Rsolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Rsolver.setIterationNumber(max_iter) 
        self.Rsolver.setTolerances(rtol=rel_tol)
        self.Rsolver.setErrorIfNotConverged(True)
        self.Rsolver.setInitialGuessNonzero(False)
        self.Rsolver.setOperators(self.R)
