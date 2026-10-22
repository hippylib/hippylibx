# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

# BiLaplacian prior on a blocked (vector-valued) parameter space

import dolfinx as dlx
from mpi4py import MPI
import sys
import os
import warnings
import dolfinx.fem.petsc
import petsc4py
import unittest
import numpy as np

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx


class Testing_Execution(unittest.TestCase):
    def setUp(self):
        comm = MPI.COMM_WORLD
        msh = dlx.mesh.create_unit_square(comm, 10, 10)
        self.Vh = dlx.fem.functionspace(msh, ("Lagrange", 1, (3,)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.prior = hpx.BiLaplacianPrior(self.Vh, 0.03, 0.3, mean=None)

    def test_vector_sizes(self):
        bs = self.Vh.dofmap.index_map_bs
        n_global = self.Vh.dofmap.index_map.size_global * bs
        m = self.prior.generate_parameter(0)
        self.assertEqual(m.petsc_vec.getSize(), n_global)
        self.assertEqual(self.prior.mean.petsc_vec.getSize(), n_global)
        self.assertEqual(self.prior.R.getSize(), (n_global, n_global))
        noise = self.prior.generate_parameter("noise")
        self.assertEqual(noise.petsc_vec.getSize(), self.prior.sqrtM.getSize()[1])

    def test_vector_mass_matrix(self):
        x = self.prior.generate_parameter(0)
        hpx.parRandom.replay()
        hpx.parRandom.normal(1.0, x)
        tmp = self.prior.generate_parameter("noise")
        y1 = self.prior.generate_parameter(0)
        y2 = self.prior.generate_parameter(0)
        self.prior.sqrtM.multTranspose(x.petsc_vec, tmp.petsc_vec)
        self.prior.sqrtM.mult(tmp.petsc_vec, y1.petsc_vec)
        self.prior.M.mult(x.petsc_vec, y2.petsc_vec)
        y2.petsc_vec.axpy(-1.0, y1.petsc_vec)
        self.assertLessEqual(y2.petsc_vec.norm(petsc4py.PETSc.NormType.N2), 1e-10)

    def test_vector_cost_grad(self):
        m = self.prior.generate_parameter(0)
        dm = self.prior.generate_parameter(0)
        hpx.parRandom.normal(1.0, m)
        hpx.parRandom.normal(1.0, dm)
        g = self.prior.generate_parameter(0)
        c0 = self.prior.cost(m)
        self.prior.grad(m, g)
        gdm = g.petsc_vec.dot(dm.petsc_vec)
        # cost is quadratic: central differences are exact up to round-off
        eps = 1e-3
        mp = self.prior.generate_parameter(0)
        mp.petsc_vec.axpy(1.0, m.petsc_vec)
        mp.petsc_vec.axpy(eps, dm.petsc_vec)
        mm = self.prior.generate_parameter(0)
        mm.petsc_vec.axpy(1.0, m.petsc_vec)
        mm.petsc_vec.axpy(-eps, dm.petsc_vec)
        fd = (self.prior.cost(mp) - self.prior.cost(mm)) / (2 * eps)
        self.assertGreater(c0, 0.0)
        self.assertLessEqual(abs(fd - gdm), 1e-8 * abs(gdm))

    def test_vector_sample(self):
        noise = self.prior.generate_parameter("noise")
        s = self.prior.generate_parameter(0)
        hpx.parRandom.normal(1.0, noise)
        self.prior.sample(noise, s)
        self.assertTrue(np.isfinite(s.petsc_vec.norm()))
        self.assertGreater(s.petsc_vec.norm(), 0.0)


if __name__ == "__main__":
    unittest.main()
