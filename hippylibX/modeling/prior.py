# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfinx as dlx
import ufl
import numpy as np
import scipy.linalg as scila
import math
from .variables import STATE, PARAMETER, ADJOINT
import numbers
import petsc4py
from mpi4py import MPI

# self.R = _BilaplacianR(self.A, self.Msolver)      
class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, A, Msolver):
        self.A = A #should be petsc4py.PETSc.Mat
        self.Msolver = Msolver  

        #help1 and help2 are vectors that can be y = (self.A)x
        #how to create them to be parallely shared by the processes.


        # self.help1, self.help2 = self.A.duplicate(), self.A.duplicate()
        # self.help1 = dlx.la.create_petsc_vector(self.A.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        # self.help2 = dlx.la.create_petsc_vector(self.A.Vh[STATE].dofmap.index_map, self.Vh[STATE].dofmap.index_map_bs) 
        
        # self.A.init_vector(self.help1, 0)
        # self.A.init_vector(self.help2, 1)
        
        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()
        

    def init_vector(self,x, dim):
        # self.A.init_vector(x,1)
        x = self.A.createVecRight() 
        #will the x be reflected outside the function call?
        return x


    def mpi_comm(self):
        # return self.A.mpi_comm()
        return self.A.comm
        
    # self.R.mult(d,Rd)
    def mult(self,x,y):
        self.A.mult(x,self.help1)
        # self.A.mult(x, self.help1)
        # self.Msolver.solve(self.help2, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)
        
class _BilaplacianRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M
        
        # self.help1, self.help2 = dl.Vector(self.M.mpi_comm()), dl.Vector(self.M.mpi_comm())
        # self.init_vector(self.help1, 0)
        # self.init_vector(self.help2, 0)
        
        # self.help1 = self.M.createVecLeft()
        # self.help2 = self.M.createVecLeft()
        self.help1, self.help2 = self.init_vector(1), self.init_vector(1)

    # def init_vector(self,x, dim):
    def init_vector(self,dim):        
        # self.M.init_vector(x,1)
        if(dim == 0):
            x = self.M.createVecLeft()
        else:
            x = self.M.createVecRight() 
        #will the x be reflected outside the function call?
        return x
        
    def solve(self,x,b):
        # nit = self.Asolver.solve(self.help1, b)
        nit = self.Asolver.solve(b, self.help1)
        self.M.mult(self.help1, self.help2)
        
        # nit += self.Asolver.solve(x, self.help2)
        nit += self.Asolver.solve(self.help2, x)

        return nit


class test_prior:
    def __init__(self, Vh, sqrt_precision_varf_handler, mean=None, rel_tol=1e-12, max_iter=1000):
        """
        Construct the prior model.
        Input:
        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """

        """
        This class implement a prior model with covariance matrix
        :math:`C = A^{-1} M A^-1`,
        where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
        """

        self.Vh = Vh
        self.sqrt_precision_varf_handler = sqrt_precision_varf_handler

        trial = ufl.TrialFunction(Vh)
        test  = ufl.TestFunction(Vh)
        
        varfM = ufl.inner(trial,test)*ufl.dx       
        self.M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        self.M.assemble()

        # self.Msolver = PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver = petsc4py.PETSc.KSP().create()
        self.Msolver.getPC().setType(petsc4py.PETSc.PC.Type.JACOBI)
        self.Msolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Msolver.setIterationNumber(max_iter) #these values should be supplied as arguments.
        self.Msolver.setTolerances(rtol=rel_tol)
        self.Msolver.setErrorIfNotConverged(True)
        self.Msolver.setInitialGuessNonzero(False)        
        self.Msolver.setOperators(self.M)

        # self.Msolver.parameters["maximum_iterations"] = max_iter
        # self.Msolver.parameters["relative_tolerance"] = rel_tol
        # self.Msolver.parameters["error_on_nonconvergence"] = True
        # self.Msolver.parameters["nonzero_initial_guess"] = False
        
        self.A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(sqrt_precision_varf_handler(trial, test) ))        
        self.A.assemble()
        self.Asolver = petsc4py.PETSc.KSP().create()
        self.Asolver.getPC().setType(petsc4py.PETSc.PC.Type.GAMG)
        self.Asolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Asolver.setIterationNumber(max_iter) #these values should be supplied as arguments.
        self.Asolver.setTolerances(rtol=rel_tol)
        self.Asolver.setErrorIfNotConverged(True)
        self.Asolver.setInitialGuessNonzero(False)        
        self.Asolver.setOperators(self.A)

        # self.Asolver = PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", amg_method())
        # self.Asolver.set_operator(self.A)
        # self.Asolver.parameters["maximum_iterations"] = max_iter
        # self.Asolver.parameters["relative_tolerance"] = rel_tol
        # self.Asolver.parameters["error_on_nonconvergence"] = True
        # self.Asolver.parameters["nonzero_initial_guess"] = False
        
        # old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        # dl.parameters["form_compiler"]["quadrature_degree"] = -1

        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}

        # representation_old = dl.parameters["form_compiler"]["representation"]
        # dl.parameters["form_compiler"]["representation"] = "quadrature"

        # num_sub_spaces = Vh.num_sub_spaces()

        num_sub_spaces = Vh.num_sub_spaces
        
        if num_sub_spaces <= 1: #SCALAR PARAMETER
            # element = ufl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
            element = ufl.FiniteElement("Quadrature", Vh.mesh.ufl_cell(), qdegree, quad_scheme="default")

        else: #Vector FIELD PARAMETER
            # element = ufl.VectorElement("Quadrature", Vh.mesh().ufl_cell(),
                                    #    qdegree, dim=num_sub_spaces, quad_scheme="default")

            element = ufl.VectorElement("Quadrature", Vh.mesh.ufl_cell(),
                                       qdegree, dim=num_sub_spaces, quad_scheme="default")

        # Qh = dl.FunctionSpace(Vh.mesh(), element)
        Qh = dlx.fem.FunctionSpace(Vh.mesh, element)

        # ph = dl.TrialFunction(Qh)
        # qh = dl.TestFunction(Qh)

        ph = ufl.TrialFunction(Qh)
        qh = ufl.TestFunction(Qh)
        
        # Mqh = dl.assemble(ufl.inner(ph,qh)*ufl.dx(metadata=metadata))
        Mqh = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,qh)*ufl.dx(metadata=metadata)) )
        Mqh.assemble()
    
        #one_constant not needed
        # if num_sub_spaces <= 1:
        #     # one_constant = dl.Constant(1.)
        #     one_constant = dlx.fem.Constant(Vh.mesh,petsc4py.PETSc.ScalarType(1.))

        # else:
        #     # one_constant = dl.Constant( tuple( [1.]*num_sub_spaces) )
        #     one_constant = dlx.fem.Constant( Vh.mesh, petsc4py.PETSc.ScalarType(tuple( [1.]*num_sub_spaces) ))

        # ones = dl.interpolate(one_constant, Qh).vector()
        # dMqh = Mqh*ones
        # Mqh.zero()
        # dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        # Mqh.set_diagonal(dMqh)
            
        ones = Mqh.createVecRight()
        ones.set(1.)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones,dMqh)
        dMqh.setArray(ones.getArray()/np.sqrt(dMqh.getArray()))
        Mqh.setDiagonal(dMqh)

        # MixedM = dl.assemble(ufl.inner(ph,test)*ufl.dx(metadata=metadata))
        MixedM = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,test)*ufl.dx(metadata=metadata)))
        MixedM.assemble()

        self.sqrtM = MixedM.matMult(Mqh)

        # nrows, ncols = self.sqrtM.getSize()

        # dense_array = np.zeros((nrows, ncols), dtype=float)        
        # row_indices, col_indices, csr_values = self.sqrtM.getValuesCSR()

        # for i in range(nrows):
        #     start = row_indices[i]
        #     end = row_indices[i + 1]
        #     dense_array[i, col_indices[start:end]] = csr_values[start:end]

        # print(np.max(dense_array),np.min(dense_array))

        # dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        # dl.parameters["form_compiler"]["representation"] = representation_old
                             
        self.R = _BilaplacianR(self.A, self.Msolver)      
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        
        self.mean = mean
        
        if self.mean is None:
            # self.mean = dl.Vector(self.R.mpi_comm())
            # self.init_vector(self.mean, 0)
            
            self.mean = self.init_vector(0)

    # def init_vector(self,x,dim):
    def init_vector(self,dim):            
        
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.
        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """

        if dim == "noise":
            # self.sqrtM.init_vector(x, 1)
            return self.sqrtM.createVecRight()
        else:
            # self.A.init_vector(x,dim)   
            if(dim == 0):
                return self.A.createVecLeft()
            else:
                return self.A.createVecRight()


    #need to construct sqrtM and Asolver from the prior in hippylib

    def sample(self, noise, s, add_mean=True):
        
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """

        rhs = self.sqrtM*noise

        # rhs = self.sqrtM.createVecLeft()
        # self.sqrtM.mult(noise,rhs)
        
        # self.Asolver.solve(s, rhs)
        # print(rhs.min(),":",rhs.max())

        #this method not giving same results in serial and parallel.
        self.Asolver.solve(rhs,s)
        # print(s.min(),":",s.max())
        

        # trial = ufl.TrialFunction(self.Vh)
        # test  = ufl.TestFunction(self.Vh)

        # dlx.fem.p9etsc.apply_lifting(rhs,[dlx.fem.form(self.sqrt_precision_varf_handler(trial, test) )],[[]])            
        # rhs.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
                    
        # self.Asolver.solve(rhs,s)

        # print(type(rhs))
        if add_mean:
            s.axpy(1., self.mean)
        

    #from Class _Prior that SqrtPrecisionPDE_Prior derives methods from
    #prior.cost is used in modelVerify in model.cost
    def cost(self,m):
        d = self.mean.copy()
        d.axpy(-1., m)

        # Rd = dl.Vector(self.R.mpi_comm())
        # self.init_vector(Rd,0)    
        # Rd = self.R.createVecLeft()

        Rd = self.init_vector(0)
        self.R.mult(d,Rd)

        # return .5*Rd.inner(d)
        # return .5*Rd.dot(d)
        loc_cost = .5*Rd.dot(d)

        return self.Vh.mesh.comm.allreduce(loc_cost, op=MPI.SUM )

def BiLaplacianPrior(Vh, gamma, delta, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False):
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior`  with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.
    
    The magnitude of :math:`\\delta\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation lenght.
    
    Here :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.
    
    Input:

    - :code:`Vh`:              the finite element space for the parameter
    - :code:`gamma` and :code:`delta`: the coefficient in the PDE (floats, dl.Constant, dl.Expression, or dl.Function)
    - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
    - :code:`mean`:            the prior mean
    - :code:`rel_tol`:         relative tolerance for solving linear systems involving covariance matrix
    - :code:`max_iter`:        maximum number of iterations for solving linear systems involving covariance matrix
    - :code:`robin_bc`:        whether to use Robin boundary condition to remove boundary artifacts
    """
    # if isinstance(gamma, numbers.Number):
        # gamma = dl.Constant(gamma)
        # gamma = dlx.fem.Constant()
        # gamma = dlx.fem.Constant( Vh.mesh, petsc4py.PETSc.ScalarType(tuple( [1.]*num_sub_spaces) ))

    # if isinstance(delta, numbers.Number):
        # delta = dl.Constant(delta)

    
    def sqrt_precision_varf_handler(trial, test): 
        if Theta == None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
        else:
            varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx
        
        varfM = ufl.inner(trial,test)*ufl.dx
        
        varf_robin = ufl.inner(trial,test)*ufl.ds
        
        if robin_bc:
            # robin_coeff = gamma*ufl.sqrt(delta/gamma)/dl.Constant(1.42)
            robin_coeff = gamma*ufl.sqrt(delta/gamma)/1.42
            
        else:
            # robin_coeff = dl.Constant(0.)
            robin_coeff = 0.

        return gamma*varfL + delta*varfM + robin_coeff*varf_robin

    # return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter)
    return test_prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter)
