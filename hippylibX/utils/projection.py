import dolfinx as dlx
import ufl
import petsc4py


def projection(v, target_func, bcs=[]):
    # reference:
    # https://github.com/michalhabera/dolfiny/blob/master/dolfiny/projection.py

    # v -> expression to project
    # target_func -> function that contains the projection

    V = target_func.function_space
    dx = ufl.dx(V.mesh)
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dlx.fem.form(ufl.inner(Pv, w) * dx)
    L = dlx.fem.form(ufl.inner(v, w) * dx)

    bcs = []
    A = dlx.fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = dlx.fem.petsc.assemble_vector(L)
    dlx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
    )

    solver = petsc4py.PETSc.KSP().create()
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
