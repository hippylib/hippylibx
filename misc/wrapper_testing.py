import sys
import os
import numpy as np
from mpi4py import MPI
import dolfinx as dlx
import ufl
import petsc4py

sys.path.append(os.path.abspath("../"))

import hippylibX as hpx

sys.path.append(os.path.abspath("../example"))


# decorator for functions in classes that are not used -> may not be needed in the final
# version of X
def unused_function(func):
    return None


comm = MPI.COMM_WORLD
rank = comm.rank
nproc = comm.size
nx = 10
ny = 10


def generate_parameter(Vh) -> dlx.la.Vector:
    return dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)


def _createsolver(Vh, petsc_options) -> petsc4py.PETSc.KSP:
    ksp = petsc4py.PETSc.KSP().create(Vh.mesh.comm)
    problem_prefix = "dolfinx_solve_prior"
    ksp.setOptionsPrefix(problem_prefix)

    # Set PETSc options
    opts = petsc4py.PETSc.Options()
    opts.prefixPush(problem_prefix)
    # petsc options for solver

    # Example:
    if petsc_options is not None:
        for k, v in petsc_options.items():
            opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    return ksp


def sqrt_precision_varf_handler(
    trial: ufl.TrialFunction, test: ufl.TestFunction
) -> ufl.form.Form:
    gamma = 0.1
    delta = 1.0

    varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx(
        metadata={"quadrature_degree": 4}
    )

    varfM = ufl.inner(trial, test) * ufl.dx

    return gamma * varfL + delta * varfM


class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):
        self.A = A  # should be petsc4py.PETSc.Mat
        self.Msolver = Msolver

        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

    @unused_function
    def init_vector(self, dim: int) -> petsc4py.PETSc.Vec:
        x = self.A.createVecRight()
        return x

    def mpi_comm(self):
        return self.A.comm

    def mult(self, mat, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


# created solely for testing if wrapper and w/o wrapper give same results
# R_org_without_wrapper is of this type.
class _BilaplacianR_wo_wrapper:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, A: petsc4py.PETSc.Mat, Msolver: petsc4py.PETSc.KSP):
        self.A = A  # should be petsc4py.PETSc.Mat
        self.Msolver = Msolver

        self.help1 = self.A.createVecLeft()
        self.help2 = self.A.createVecRight()

    @unused_function
    def init_vector(self, dim: int) -> petsc4py.PETSc.Vec:
        x = self.A.createVecRight()
        return x

    def mpi_comm(self):
        return self.A.comm

    # when using wrapper
    # mat needed for petsc-wrapper call.
    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help1, self.help2)
        self.A.mult(self.help2, y)


# created solely for testing if wrapper and w/o wrapper give same results
# Rsolver_org is of this type.
class _BilaplacianRsolver:
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """

    def __init__(self, Asolver: petsc4py.PETSc.KSP, M: petsc4py.PETSc.Mat):
        self.Asolver = Asolver
        self.M = M
        self.help1, self.help2 = self.M.createVecLeft(), self.M.createVecLeft()

    @unused_function
    def init_vector(self, dim):
        if dim == 0:
            x = self.M.createVecLeft()
        else:
            x = self.M.createVecRight()
        return x

    def solve(self, x: petsc4py.PETSc.Vec, b: petsc4py.PETSc.Vec):
        self.Asolver.solve(b, self.help1)
        nit = self.Asolver.its
        self.M.mult(self.help1, self.help2)
        self.Asolver.solve(self.help2, x)
        nit += self.Asolver.its
        return nit


# main starts here

msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 2))
Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))

trial = ufl.TrialFunction(Vh_m)
test = ufl.TestFunction(Vh_m)

dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

# creating M and Msolver
petsc_options_M = {
    "ksp_type": "cg",
    "pc_type": "jacobi",
    "ksp_rtol": "1e-12",
    "ksp_max_it": "1000",
    "ksp_error_if_not_converged": "true",
    "ksp_initial_guess_nonzero": "false",
}
varfM = ufl.inner(trial, test) * dx
M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
M.assemble()
Msolver = _createsolver(Vh_m, petsc_options_M)
Msolver.setOperators(M)

# creating A and Asolver
petsc_options_A = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": "1e-12",
    "ksp_max_it": "1000",
    "ksp_error_if_not_converged": "true",
    "ksp_initial_guess_nonzero": "false",
}
A = dlx.fem.petsc.assemble_matrix(
    dlx.fem.form(sqrt_precision_varf_handler(trial, test))
)
A.assemble()
Asolver = _createsolver(Vh_m, petsc_options_A)
if petsc_options_A["pc_type"] == "hypre":
    pc = Asolver.getPC()
    pc.setHYPREType("boomeramg")
Asolver.setOperators(A)

R_org = _BilaplacianR(A, Msolver)


# two objects for producing results without wrappers
R_org_wo_wrapper = _BilaplacianR_wo_wrapper(A, Msolver)
Rsolver_org = _BilaplacianRsolver(Asolver, M)

# #dummy vectors:
vec1 = generate_parameter(Vh_m)
vec2_without_wrapper = generate_parameter(Vh_m)
vec2_with_wrapper = generate_parameter(Vh_m)


hpx.parRandom.normal(1.0, vec1)

# attempt with wrapper
# for R
R = petsc4py.PETSc.Mat().createPython(A.getSizes(), comm=Vh_m.mesh.comm)
R.setPythonContext(R_org)
R.setUp()

# test for R:
# R.mult(dlx.la.create_petsc_vector_wrap(vec1), dlx.la.create_petsc_vector_wrap(vec2_with_wrapper))
# R_org_wo_wrapper.mult(dlx.la.create_petsc_vector_wrap(vec1), dlx.la.create_petsc_vector_wrap(vec2_without_wrapper))

# test for Rsolver:
petsc_options_Rsolver = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": "1e-12",
    "ksp_max_it": "1000",
    "ksp_error_if_not_converged": "true",
    "ksp_initial_guess_nonzero": "false",
}
Rsolver = _createsolver(Vh_m, petsc_options_Rsolver)
if petsc_options_Rsolver["pc_type"] == "hypre":
    pc = Rsolver.getPC()
    pc.setHYPREType("boomeramg")

Rsolver.setOperators(R)

Rsolver.solve(
    dlx.la.create_petsc_vector_wrap(vec1),
    dlx.la.create_petsc_vector_wrap(vec2_with_wrapper),
)  # works
Rsolver_org.solve(
    dlx.la.create_petsc_vector_wrap(vec2_without_wrapper),
    dlx.la.create_petsc_vector_wrap(vec1),
)


# ***********************************************

# w/o and with using wrapper

comm = Vh_m.mesh.comm
test_func = hpx.vector2Function(vec2_without_wrapper, Vh_m)

V = test_func.function_space

x_coords, y_coords = dlx.fem.Function(V), dlx.fem.Function(V)
x_coords.interpolate(lambda x: x[0])
y_coords.interpolate(lambda x: x[1])

x_coords_loc = x_coords.vector.array
y_coords_loc = y_coords.vector.array

u_at_vertices = dlx.fem.Function(V)
u_at_vertices.interpolate(test_func)
values_loc = u_at_vertices.vector.array

loc_vals_comb = [x_coords_loc, y_coords_loc, values_loc]

all_vals_comb = comm.gather(loc_vals_comb, root=0)

if comm.rank == 0:
    all_x, all_y, all_vals = [], [], []

    for sublist in all_vals_comb:
        all_x.append(sublist[0])
        all_y.append(sublist[1])
        all_vals.append(sublist[2])

    all_x_flat = np.concatenate([arr.flatten() for arr in all_x])
    all_y_flat = np.concatenate([arr.flatten() for arr in all_y])
    all_vals_flat = np.concatenate([arr.flatten() for arr in all_vals])

    combined_tuple_version = [
        (all_x_flat[i], all_y_flat[i], all_vals_flat[i]) for i in range(len(all_x_flat))
    ]

    sorted_combined_tuple_version = sorted(
        combined_tuple_version, key=lambda x: (x[0], x[1])
    )

    x_coords_final, y_coords_final, values_final = [], [], []

    for j in range(len(sorted_combined_tuple_version)):
        x_coords_final.append(sorted_combined_tuple_version[j][0])
        y_coords_final.append(sorted_combined_tuple_version[j][1])
        values_final.append(sorted_combined_tuple_version[j][2])

    np.savetxt(f"x_coords_X_{comm.size}_procs_without_wrapper", x_coords_final)
    np.savetxt(f"y_coords_X_{comm.size}_procs_without_wrapper", y_coords_final)
    np.savetxt(f"R_results_{comm.size}_procs_without_wrapper", values_final)


comm = Vh_m.mesh.comm
test_func = hpx.vector2Function(vec2_with_wrapper, Vh_m)

V = test_func.function_space

x_coords, y_coords = dlx.fem.Function(V), dlx.fem.Function(V)
x_coords.interpolate(lambda x: x[0])
y_coords.interpolate(lambda x: x[1])

x_coords_loc = x_coords.vector.array
y_coords_loc = y_coords.vector.array

u_at_vertices = dlx.fem.Function(V)
u_at_vertices.interpolate(test_func)
values_loc = u_at_vertices.vector.array

loc_vals_comb = [x_coords_loc, y_coords_loc, values_loc]

all_vals_comb = comm.gather(loc_vals_comb, root=0)

if comm.rank == 0:
    all_x, all_y, all_vals = [], [], []

    for sublist in all_vals_comb:
        all_x.append(sublist[0])
        all_y.append(sublist[1])
        all_vals.append(sublist[2])

    all_x_flat = np.concatenate([arr.flatten() for arr in all_x])
    all_y_flat = np.concatenate([arr.flatten() for arr in all_y])
    all_vals_flat = np.concatenate([arr.flatten() for arr in all_vals])

    combined_tuple_version = [
        (all_x_flat[i], all_y_flat[i], all_vals_flat[i]) for i in range(len(all_x_flat))
    ]

    sorted_combined_tuple_version = sorted(
        combined_tuple_version, key=lambda x: (x[0], x[1])
    )

    x_coords_final, y_coords_final, values_final = [], [], []

    for j in range(len(sorted_combined_tuple_version)):
        x_coords_final.append(sorted_combined_tuple_version[j][0])
        y_coords_final.append(sorted_combined_tuple_version[j][1])
        values_final.append(sorted_combined_tuple_version[j][2])

    np.savetxt(f"x_coords_X_{comm.size}_procs_with_wrapper", x_coords_final)
    np.savetxt(f"y_coords_X_{comm.size}_procs_with_wrapper", y_coords_final)
    np.savetxt(f"R_results_{comm.size}_procs_with_wrapper", values_final)

with_wrapper = np.loadtxt(f"R_results_{comm.size}_procs_with_wrapper")
without_wrapper = np.loadtxt(f"R_results_{comm.size}_procs_without_wrapper")

if comm.rank == 0:
    print(np.linalg.norm(with_wrapper - without_wrapper, 2))
    print(np.linalg.norm(with_wrapper - without_wrapper, np.inf))
