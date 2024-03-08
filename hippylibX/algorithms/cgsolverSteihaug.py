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
from ..utils.parameterList import ParameterList
import math
import mpi4py
from ..utils import vector2Function
from .linalg import inner

def CGSolverSteihaug_ParameterList():
    """
    Generate a :code:`ParameterList` for :code:`CGSolverSteihaug`.
    Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
    parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
    parameters["max_iter"]      = [1000, "the maximum number of iterations"]
    parameters["zero_initial_guess"] = [True, "if True we start with a 0 initial guess; if False we use the x as initial guess."]
    parameters["print_level"] = [0, "verbosity level: -1 --> no output on screen; 0 --> only final residual at convergence or reason for not not convergence"]
    return ParameterList(parameters)

class CGSolverSteihaug:
    """
    Solve the linear system :math:`A x = b` using preconditioned conjugate gradient ( :math:`B` preconditioner)
    and the Steihaug stopping criterion:

    - reason of termination 0: we reached the maximum number of iterations (no convergence)
    - reason of termination 1: we reduced the residual up to the given tolerance (convergence)
    - reason of termination 2: we reached a negative direction (premature termination due to not spd matrix)
    - reason of termination 3: we reached the boundary of the trust region
    
    The stopping criterion is based on either

    - the absolute preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}} < atol`
    - the relative preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}}/|| r^0 ||_{B^{-1}} < rtol,`

    where :math:`r^* = b - Ax^*` is the residual at convergence and :math:`r^0 = b - Ax^0` is the initial residual.
    
    The operator :code:`A` is set using the method :code:`set_operator(A)`.
    :code:`A` must provide the following two methods:

    - :code:`A.mult(x,y)`: `y = Ax`
    - :code:`A.init_vector(x, dim)`: initialize the vector `x` so that it is compatible with the range `(dim = 0)` or
      the domain `(dim = 1)` of :code:`A`.
      
    The preconditioner :code:`B` is set using the method :code:`set_preconditioner(B)`.
    :code:`B` must provide the following method:
    - :code:`B.solve(z,r)`: `z` is the action of the preconditioner :code:`B` on the vector `r`
    
    To solve the linear system :math:`Ax = b` call :code:`self.solve(x,b)`.
    Here :code:`x` and :code:`b` are assumed to be :code:`dolfin.Vector` objects.
    
    Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default parameters and their descriptions
    """

    reason = ["Maximum Number of Iterations Reached",
              "Relative/Absolute residual less than tol",
              "Reached a negative direction",
              "Reached trust region boundary"
              ]
    def __init__(self, parameters=CGSolverSteihaug_ParameterList(),comm = mpi4py.MPI.Intracomm):
        
        self.parameters = parameters
        
        self.A = None
        self.B_solver = None
        self.B_op = None
        self.converged = False
        self.iter = 0
        self.reasonid = 0
        self.final_norm = 0

        self.TR_radius_2 = None

        self.update_x = self.update_x_without_TR
        
        self.comm = comm
        
    def set_operator(self, A):
        """
        Set the operator :math:`A`.
        """
        
        self.A = A
        self.r = self.A.init_vector(0)
        self.z = self.A.init_vector(0)
        self.d = self.A.init_vector(0)
        self.Ad = self.A.init_vector(0)
                
        
    def set_preconditioner(self, B_solver):
        """
        Set the preconditioner :math:`B`.
        """
        self.B_solver = B_solver

    def set_TR(self,radius,B_op):
        assert self.parameters["zero_initial_guess"]
        self.TR_radius_2 = radius*radius
        self.update_x = self.update_x_with_TR
        self.B_op = B_op
        self.B_op.init_vector(self.Bx,0)

    def update_x_without_TR(self,x : dlx.la.Vector, alpha : float, d : dlx.la.Vector):
        x.array[:] = x.array + alpha * d.array

        return False

    def update_x_with_TR(self,x,alpha,d):
        x_bk = x.copy()
        x.axpy(alpha,d)
        self.Bx.zero()
        self.B_op.mult(x, self.Bx)
        x_Bnorm2 = self.Bx.inner(x)

        if x_Bnorm2 < self.TR_radius_2:
            return  False
        else:
            # Move point to boundary of trust region
            self.Bx.zero()
            self.B_op.mult(x_bk, self.Bx)
            x_Bnorm2 = self.Bx.inner(x_bk)
            Bd = self.d.copy()
            Bd.zero()
            self.B_op.mult(self.d,Bd)
            d_Bnorm2 = Bd.inner(d)
            d_Bx = Bd.inner(x_bk)
            a_tau = alpha*alpha*d_Bnorm2
            b_tau_half = alpha* d_Bx
            c_tau = x_Bnorm2- self.TR_radius_2
            # Solve quadratic for :code:`tau`
            tau = (-b_tau_half + math.sqrt(b_tau_half*b_tau_half - a_tau*c_tau))/a_tau
            x.zero()
            x.axpy(1,x_bk)
            x.axpy(tau*alpha, d)

            return  True
        
    def solve(self,x,b):
        """
        Solve the linear system :math:`Ax = b`
        """

        self.iter = 0
        self.converged = False
        self.reasonid  = 0
        
        betanom = 0.0
        alpha = 0.0 
        beta = 0.0


        if self.parameters["zero_initial_guess"]:
            self.r.array[:] = 0.
            self.r.array[:] = self.r.array + 1. * b.array
            x.array[:] = 0.
                        
        else:
            assert self.TR_radius_2==None
            self.A.mult(x,self.r)
            self.r.array[:] = -1. * self.r.array
            self.r.array[:] = self.r.array + 1. * b.array

        self.z.array[:] = 0.

        self.B_solver.solve(self.z,self.r) #z = B^-1 r  
              
        self.d.array[:] = 0.

        self.d.array[:] = self.d.array + 1. * self.z.array

        nom0 = inner(self.d,self.r)

        nom = nom0
        
        if self.parameters["print_level"] == 1:
            print(" Iterartion : ", 0, " (B r, r) = ", nom)
            
        rtol2 = nom * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
        atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
        r0 = max(rtol2, atol2)
        
        if nom <= r0:
            self.converged  = True
            self.reasonid   = 1
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print( self.reason[self.reasonid])
                print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
            
            return
        

        self.A.mult(self.d, self.Ad)
        den = inner(self.Ad, self.d)
        if den <= 0.0:
            self.converged = True
            self.reasonid = 2
            x.array[:] = x.array + 1. * self.d.array
            self.r.array[:] = self.r.array + (-1.) * self.Ad.array
            self.B_solver.solve(self.z, self.r)

            nom = inner(self.r, self.z)
            self.final_norm = math.sqrt(nom)
            if(self.parameters["print_level"] >= 0):
                print( self.reason[self.reasonid])
                print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
            
            return
        
        # start iteration
        self.iter = 1
        while True:
            alpha = nom/den
            TrustBool = self.update_x(x,alpha,self.d)   # x = x + alpha d
            if TrustBool == True:
                self.converged = True
                self.reasonid = 3
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print( self.reason[self.reasonid] )
                    print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
                break

            self.r.array[:]  = self.r.array + (-alpha) * self.Ad.array\
            
            self.B_solver.solve(self.z, self.r)     # z = B^-1 r
            betanom = inner(self.r, self.z)
            if self.parameters["print_level"] == 1:
                print( " Iteration : ", self.iter, " (B r, r) = ", betanom)

            
            if betanom < r0:
                self.converged = True
                self.reasonid = 1
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print( self.reason[self.reasonid])
                    print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
                break

    
            self.iter += 1
            if self.iter > self.parameters["max_iter"]:
                self.converged = False
                self.reasonid = 0
                self.final_norm = math.sqrt(betanom)
                if(self.parameters["print_level"] >= 0):
                    print( self.reason[self.reasonid] )
                    print( "Not Converged. Final residual norm ", self.final_norm )
                break
        
            beta = betanom/nom
            self.d.array[:] = beta * self.d.array
            self.d.array[:] = self.d.array + 1. * self.z.array
            self.A.mult(self.d,self.Ad)
                
            den = inner(self.d, self.Ad)
            if den <= 0.0:
                self.converged = True
                self.reasonid = 2
                self.final_norm = math.sqrt(nom)
                if(self.parameters["print_level"] >= 0):
                    print( self.reason[self.reasonid] )
                    print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm )
                break
            
            nom = betanom       