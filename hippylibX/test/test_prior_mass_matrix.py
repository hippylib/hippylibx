# x -> random dolfinx Vector
# in modeling/prior.py:
# sqrtM . sqrtM.T . x == M . x

import ufl
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import sys
import os
import dolfinx.fem.petsc
from matplotlib import pyplot as plt
from typing import Sequence, Dict
import petsc4py
import unittest


# import ufl.sobolevspace
import basix.ufl

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx

class Testing_Execution(unittest.TestCase):
    def test_prior_mass_matrix(self):
        nx, ny = 10, 10

        comm = MPI.COMM_WORLD

        msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
        Vh = dlx.fem.functionspace(msh, ("Lagrange", 1))

        qdegree = 2 * Vh._ufl_element.degree
        metadata = {"quadrature_degree": qdegree}

        element = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), degree = qdegree)        

        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)

        Qh = dlx.fem.functionspace(Vh.mesh, element)

        ph = ufl.TrialFunction(Qh)
        qh = ufl.TestFunction(Qh)


        Mqh = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, qh) * ufl.dx(metadata=metadata))
        )
        Mqh.assemble()

        ones = Mqh.createVecRight()
        ones.set(1.0)
        dMqh = Mqh.createVecLeft()
        Mqh.mult(ones, dMqh)
        dMqh.setArray(ones.getArray() / np.sqrt(dMqh.getArray()))
        Mqh.setDiagonal(dMqh)

        MixedM = dlx.fem.petsc.assemble_matrix(
            dlx.fem.form(ufl.inner(ph, test) * ufl.dx(metadata=metadata))
        )
        MixedM.assemble()

        sqrtM = MixedM.matMult(Mqh)

        varfM = ufl.inner(trial, test) *ufl.Measure("dx", metadata={"quadrature_degree": 4})

        M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
        M.assemble()


        x = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        hpx.parRandom.replay()
        hpx.parRandom.normal(1., x)
        y1 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
        y2 = dlx.la.vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)

        temp = sqrtM.createVecRight()

        sqrtM.multTranspose(x.petsc_vec,temp)
        sqrtM.mult(temp,y1.petsc_vec)

        M.mult(x.petsc_vec,y2.petsc_vec)

        y2.petsc_vec.axpy(-1., y1.petsc_vec)
        value = y2.petsc_vec.norm(petsc4py.PETSc.NormType.N2)
        self.assertLessEqual(
            np.abs(value),
            1e-6,
            "prior_sqrtM creation failed",
        )

if __name__ == "__main__":
    unittest.main()
