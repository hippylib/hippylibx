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
# import scipy.linalg as scila
import math
from .variables import STATE, PARAMETER, ADJOINT
import numbers
import petsc4py
from mpi4py import MPI

#decorator for functions in classes that are not used -> may not be needed in the final
#version of X

def unused_function(func):
    return None


# self.R = _BilaplacianR(self.A, self.Msolver)      
class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, A : petsc4py.PETSc.Mat, Msolver : petsc4py.PETSc.KSP):
        self.A = A #should be petsc4py.PETSc.Mat
        self.Msolver = Msolver  
        
        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()
        
    @unused_function
    def init_vector(self, dim : int) -> petsc4py.PETSc.Vec:
        x = self.A.createVecRight() 
        return x

    def mpi_comm(self):
        return self.A.comm
        
    def mult(self,x : petsc4py.PETSc.Vec, y : petsc4py.PETSc.Vec) -> None:
        self.A.mult(x,self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


class _BilaplacianRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver : petsc4py.PETSc.KSP, M : petsc4py.PETSc.Mat):
        self.Asolver = Asolver
        self.M = M
        
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    @unused_function
    def init_vector(self,dim):        
        if(dim == 0):
            x = self.M.createVecLeft()
        else:
            x = self.M.createVecRight() 
        return x
    
    def solve(self,x : dlx.la.Vector, b : dlx.la.Vector):
        
        # self.Asolver.solve(dlx.la.create_petsc_vector_wrap(b), self.help1)
        # nit = self.Asolver.its
        # self.M.mult(self.help1, self.help2)
        # self.Asolver.solve(self.help2,dlx.la.create_petsc_vector_wrap(x))
        # nit += self.Asolver.its
        # return nit

        temp_petsc_vec_b = dlx.la.create_petsc_vector_wrap(b)
        temp_petsc_vec_x = dlx.la.create_petsc_vector_wrap(x)
        self.Asolver.solve(temp_petsc_vec_b, self.help1)
        nit = self.Asolver.its
        self.M.mult(self.help1, self.help2)
        self.Asolver.solve(self.help2,temp_petsc_vec_x)
        nit += self.Asolver.its
        temp_petsc_vec_b.destroy()
        temp_petsc_vec_x.destroy()
        return nit


class test_prior:
    def __init__(self, Vh : dlx.fem.FunctionSpace, sqrt_precision_varf_handler, mean=None, rel_tol=1e-12, max_iter=1000):
        
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

        self.Msolver = petsc4py.PETSc.KSP().create()
        self.Msolver.getPC().setType(petsc4py.PETSc.PC.Type.JACOBI)
        self.Msolver.setType(petsc4py.PETSc.KSP.Type.CG)
        self.Msolver.setIterationNumber(max_iter) #these values should be supplied as arguments.
        self.Msolver.setTolerances(rtol=rel_tol)
        self.Msolver.setErrorIfNotConverged(True)
        self.Msolver.setInitialGuessNonzero(False)        
        self.Msolver.setOperators(self.M)
        
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

        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}

        num_sub_spaces = Vh.num_sub_spaces
        
        if num_sub_spaces <= 1: #SCALAR PARAMETER
            element = ufl.FiniteElement("Quadrature", Vh.mesh.ufl_cell(), qdegree, quad_scheme="default")

        else: #Vector FIELD PARAMETER
            element = ufl.VectorElement("Quadrature", Vh.mesh.ufl_cell(),
                                       qdegree, dim=num_sub_spaces, quad_scheme="default")

        self.Qh = dlx.fem.FunctionSpace(Vh.mesh, element)

        ph = ufl.TrialFunction(self.Qh)
        qh = ufl.TestFunction(self.Qh)
        
        Mqh = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,qh)*ufl.dx(metadata=metadata)) )
        Mqh.assemble()
                
        ones = Mqh.createVecRight()
        ones.set(1.)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones,dMqh)
        dMqh.setArray(ones.getArray()/np.sqrt(dMqh.getArray()))
        Mqh.setDiagonal(dMqh)

        MixedM = dlx.fem.petsc.assemble_matrix(dlx.fem.form(ufl.inner(ph,test)*ufl.dx(metadata=metadata)))
        MixedM.assemble()

        self.sqrtM = MixedM.matMult(Mqh)

        # self.sqrtM_function_space = Qh
                         
        self.R = _BilaplacianR(self.A, self.Msolver)      
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
        
        self.mean = mean
        
        if self.mean is None:
            # self.mean = dl.Vector(self.R.mpi_comm())
            # self.init_vector(self.mean, 0)
            
            self.mean = self.init_vector(0)

    #replaced function init_vector with generate_parameter
    @unused_function
    def init_vector(self, dim : int) -> dlx.la.Vector:            
        
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.
        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """

        if dim == "noise":
            return dlx.la.vector( self.sqrtM_function_space.dofmap.index_map    )

        else:
            if(dim == 0):
                return dlx.la.vector( self.Vh.dofmap.index_map)
                
            else:
                return dlx.la.vector( self.Vh.dofmap.index_map )

    #Modified version of above init_vector
    def generate_parameter(self, dim : int) -> dlx.la.Vector:      
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.
        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if(dim == "noise"):
            return dlx.la.vector( self.Qh.dofmap.index_map )
        else:
            return dlx.la.vector( self.Vh.dofmap.index_map )

    #need to construct sqrtM and Asolver from the prior in hippylib
    def sample(self, noise : dlx.la.Vector, s : dlx.la.Vector, add_mean=True) -> None:

        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        temp_petsc_vec_noise = dlx.la.create_petsc_vector_wrap(noise)

        # rhs = self.sqrtM*dlx.la.create_petsc_vector_wrap(noise)
        rhs = self.sqrtM*temp_petsc_vec_noise
        temp_petsc_vec_noise.destroy()

        # s_petsc = dlx.la.create_petsc_vector_wrap(s)
        # self.Asolver.solve(rhs,s_petsc)
        
        temp_petsc_vec_s = dlx.la.create_petsc_vector_wrap(s)
        self.Asolver.solve(rhs,temp_petsc_vec_s)

        # s_petsc.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        # s_petsc.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)
        
        temp_petsc_vec_s.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
        temp_petsc_vec_s.ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

        if add_mean:
            # s_petsc.axpy(1., dlx.la.create_petsc_vector_wrap(self.mean))
            temp_petsc_vec_mean = dlx.la.create_petsc_vector_wrap(self.mean)
            temp_petsc_vec_s.axpy(1., temp_petsc_vec_mean)
            temp_petsc_vec_mean.destroy()
        
        rhs.destroy()
        temp_petsc_vec_s.destroy()

    def cost(self,m : dlx.la.Vector) -> float:  
        # d = dlx.la.create_petsc_vector_wrap(self.mean).copy()
        # d.axpy(-1., dlx.la.create_petsc_vector_wrap(m))
        # # Rd = dlx.la.create_petsc_vector_wrap(self.init_vector(0))
        # Rd = dlx.la.create_petsc_vector_wrap(self.generate_parameter(0))
        
        # self.R.mult(d,Rd)
        # return .5*Rd.dot(d)
    
        temp_petsc_vec_d = dlx.la.create_petsc_vector_wrap(self.mean).copy()
        temp_petsc_vec_m = dlx.la.create_petsc_vector_wrap(m)
        temp_petsc_vec_d.axpy(-1., temp_petsc_vec_m)
        temp_petsc_vec_Rd = dlx.la.create_petsc_vector_wrap(self.generate_parameter(0))
        
        self.R.mult(temp_petsc_vec_d,temp_petsc_vec_Rd)
        return_value = .5*temp_petsc_vec_Rd.dot(temp_petsc_vec_d)
        temp_petsc_vec_d.destroy()
        temp_petsc_vec_m.destroy()
        temp_petsc_vec_Rd.destroy()
        return return_value

    def grad(self,m : dlx.la.Vector, out : dlx.la.Vector) -> None:
        # d = dlx.la.create_petsc_vector_wrap(m).copy()
        # d.axpy(-1., dlx.la.create_petsc_vector_wrap(self.mean))
        # self.R.mult(d,dlx.la.create_petsc_vector_wrap(out))

        temp_petsc_vec_d = dlx.la.create_petsc_vector_wrap(m).copy()
        temp_petsc_vec_self_mean = dlx.la.create_petsc_vector_wrap(self.mean)
        temp_petsc_vec_out = dlx.la.create_petsc_vector_wrap(out)


        temp_petsc_vec_d.axpy(-1., temp_petsc_vec_self_mean)


        self.R.mult(temp_petsc_vec_d,temp_petsc_vec_out)

        temp_petsc_vec_d.destroy()
        temp_petsc_vec_self_mean.destroy()
        temp_petsc_vec_out.destroy()


def BiLaplacianPrior(Vh : dlx.fem.FunctionSpace, gamma : float, delta : float, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False) -> test_prior:
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
    
    def sqrt_precision_varf_handler(trial : ufl.TrialFunction, test : ufl.TestFunction) -> ufl.form.Form: 
        if Theta == None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
        else:
            varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx
        
        varfM = ufl.inner(trial,test)*ufl.dx
        
        varf_robin = ufl.inner(trial,test)*ufl.ds
        
        if robin_bc:
            robin_coeff = gamma*ufl.sqrt(delta/gamma)/1.42
            
        else:
            robin_coeff = 0.
        
        return gamma*varfL + delta*varfM + robin_coeff*varf_robin

    return test_prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter)
