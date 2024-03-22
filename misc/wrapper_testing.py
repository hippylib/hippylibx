import unittest
import sys
import os
import numpy as np
from mpi4py import MPI
import dolfinx as dlx
import ufl
import petsc4py

sys.path.append(os.path.abspath('../'))

import hippylibX as hpx
sys.path.append(os.path.abspath('../example'))



#decorator for functions in classes that are not used -> may not be needed in the final
#version of X
def unused_function(func):
    return None

comm = MPI.COMM_WORLD
rank  = comm.rank
nproc = comm.size
nx = 10
ny = 10

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

    #when using wrapper
    #mat needed for petsc-wrapper call.         
    def mult(self, mat, x : petsc4py.PETSc.Vec, y : petsc4py.PETSc.Vec) -> None:
        self.A.mult(x,self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)

    #when not using wrapper (done to compare results for
    #with and without wrapper)
    # def mult(self, x : petsc4py.PETSc.Vec, y : petsc4py.PETSc.Vec) -> None:
    #     self.A.mult(x,self.help1)
    #     self.Msolver.solve(self.help1, self.help2)
    #     self.A.mult(self.help2, y)


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

    #when using wrapper
    def solve(self, ksp, x : dlx.la.Vector, b : dlx.la.Vector):
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

    #when not using wrapper (done to compare results for
    #with and without wrapper)
    # def solve(self, x : dlx.la.Vector, b : dlx.la.Vector):
    #     temp_petsc_vec_b = dlx.la.create_petsc_vector_wrap(b)
    #     temp_petsc_vec_x = dlx.la.create_petsc_vector_wrap(x)
    #     self.Asolver.solve(temp_petsc_vec_b, self.help1)
    #     nit = self.Asolver.its
    #     self.M.mult(self.help1, self.help2)
    #     self.Asolver.solve(self.help2,temp_petsc_vec_x)
    #     nit += self.Asolver.its
    #     temp_petsc_vec_b.destroy()
    #     temp_petsc_vec_x.destroy()
    #     return nit

def generate_parameter(Vh) -> dlx.la.Vector:
    return dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs) 

def _createsolver(Vh, petsc_options) -> petsc4py.PETSc.KSP:
    ksp = petsc4py.PETSc.KSP().create(Vh.mesh.comm)
    problem_prefix = f"dolfinx_solve_prior"
    ksp.setOptionsPrefix(problem_prefix)

    # Set PETSc options
    opts = petsc4py.PETSc.Options()
    opts.prefixPush(problem_prefix)
    #petsc options for solver
    
    #Example:
    if petsc_options is not None:
        for k, v in petsc_options.items():
            opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    return ksp

def sqrt_precision_varf_handler(trial : ufl.TrialFunction, test : ufl.TestFunction) -> ufl.form.Form: 
    gamma = 0.1
    delta = 1.

    varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx(metadata = {"quadrature_degree":4})
    
    varfM = ufl.inner(trial,test)*ufl.dx
    
    return gamma*varfL + delta*varfM 


#main starts here

msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)    
Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 2)) 
Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))

trial = ufl.TrialFunction(Vh_m)
test  = ufl.TestFunction(Vh_m)

dx = ufl.Measure("dx",metadata={"quadrature_degree":4})
ds = ufl.Measure("ds",metadata={"quadrature_degree":4})

#creating M and Msolver
petsc_options_M = {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol":"1e-12", "ksp_max_it":"1000", "ksp_error_if_not_converged":"true","ksp_initial_guess_nonzero":"false"} 
varfM = ufl.inner(trial,test)*dx   
M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
M.assemble()
Msolver = _createsolver(Vh_m, petsc_options_M)
Msolver.setOperators(M)

#creating A and Asolver
petsc_options_A = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol":"1e-12", "ksp_max_it":"1000", "ksp_error_if_not_converged":"true", "ksp_initial_guess_nonzero":"false"}
A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(sqrt_precision_varf_handler(trial, test) ))        
A.assemble()
Asolver = _createsolver(Vh_m, petsc_options_A)
if(petsc_options_A['pc_type'] == 'hypre'):
    pc = Asolver.getPC()
    pc.setHYPREType('boomeramg')
Asolver.setOperators(A)

R_org = _BilaplacianR(A, Msolver)  
Rsolver_org = _BilaplacianRsolver(Asolver, M)

loc_size, glb_size = A.getSizes()

R = petsc4py.PETSc.Mat().createPython(A.getSizes(),comm = Vh_m.mesh.comm)
R.setPythonContext(R_org)
R.setUp()

# #dummy vectors:
vec1 = generate_parameter(Vh_m)
vec2 = generate_parameter(Vh_m)

for i in range(len(vec1.array)):
    vec1.array[i] = i

petsc_vec1 = dlx.la.create_petsc_vector_wrap(vec1)
petsc_vec2 = dlx.la.create_petsc_vector_wrap(vec2)

#this works
R.mult(petsc_vec1,petsc_vec2)
# R_org.mult(petsc_vec1,petsc_vec2)

Rsolver = petsc4py.PETSc.KSP().createPython(comm = Vh_m.mesh.comm)
Rsolver.setPythonContext(Rsolver_org)
Rsolver.setUp() #error here

#this gives error
#testing Rsolver:
# Rsolver.solve(petsc_vec2 ,petsc_vec1)
# Rsolver_org.solve(vec2, vec1)