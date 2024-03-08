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


from .variables import STATE, PARAMETER, ADJOINT
import dolfinx as dlx
from ..algorithms import linalg

#decorator for functions in classes that are not used -> may not be needed in the final
#version of X
def unused_function(func):
    return None

class ReducedHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def __init__(self, model, misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.gauss_newton_approx = self.model.gauss_newton_approx 
        self.misfit_only=misfit_only
        self.ncalls = 0
        
        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat = model.generate_vector(STATE)
        self.phat = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)
    

    def init_vector(self, dim: int) -> dlx.la.Vector:
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be reshaped to be compatible with the range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be compatible with the domain of the reduced Hessian.
               
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and the domain is the same. Either way, we choosed to add the parameter :code:`dim` for consistency with the interface of :code:`Matrix` in dolfin.
        """

        return self.model.init_parameter(dim)
        

    def mult(self,x : dlx.la.Vector ,y : dlx.la.Vector ) -> None:

        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`. Return the result in :code:`y`.
        """
        if self.gauss_newton_approx:
            self.GNHessian(x,y)
        else:
            self.TrueHessian(x,y)
        
        self.ncalls += 1
    
    def inner(self, x : dlx.la.Vector, y : dlx.la.Vector):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the reduced
        Hessian :math:`H,\\,(x, y)_H = x' H y`.
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.array[:] = 0.

        self.mult(y,Ay)
        
        return linalg.inner(x,Ay)

    def GNHessian(self,x,y):
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)            
            y.array[:] = y.array + 1. * self.yhelp.array

            
        
    def TrueHessian(self, x : dlx.la.Vector , y : dlx.la.Vector) -> None:
        """
        Apply the the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """

        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)

        self.rhs_adj.array[:] = self.rhs_adj.array + (-1.) * self.rhs_adj2.array
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)

        y.array[:] = y.array + 1. * self.yhelp.array

        self.model.applyWmu(self.uhat, self.yhelp)
        y.array[:] = y.array + (-1.) * self.yhelp.array
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.array[:] = y.array + 1. * self.yhelp.array
        
 