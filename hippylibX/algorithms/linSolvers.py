import petsc4py


#I don't use either of the following 2 functions for now,
# currently just using this in the _createLUSolver method in
# the PDEVariationalProblem class in modeling/PDEProblem.py file

# def _createLUSolver(self):
        # ksp = petsc4py.PETSc.KSP().create()
        # return ksp


def _PETScLUSolver_set_operator(self, A):
    if hasattr(A, 'mat'):
        self.ksp().setOperators(A.mat())
    else:
        # self.ksp().setOperators(dl.as_backend_type(A).mat())
        self.ksp().setOperators(A.instance()) #dont know how to test this?


def PETScLUSolver(comm, method='default'):
    if not hasattr(petsc4py.PETSc.PETScLUSolver, 'set_operator'):
        petsc4py.PETSC.PETScLUSolver.set_operator = _PETScLUSolver_set_operator
    return petsc4py.PETSc.PETScLUSolver(comm, method)