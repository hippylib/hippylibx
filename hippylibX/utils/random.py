# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

from typing import Union

import petsc4py
from mpi4py import MPI

import dolfinx as dlx
import numpy as np

from ..algorithms.multivector import MultiVector


class Random:
    """
    This class handles parallel generation of random numbers in hippylibX.
    """

    def __init__(self, rank: int, nproc: int, seed=1) -> None:
        """
        Create a parallel random number number generator.

        INPUTS:
            - :code:`rank`: id of the calling process.
            - :code:`nproc`: number of processor in the communicator.
            - :code:`seed`: random seed to initialize the random engine.
        """
        seed_sequence = np.random.SeedSequence(seed)
        self.child_seeds = seed_sequence.spawn(nproc)
        self.rank = rank
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def replay(self) -> None:
        """
        Reinitialize seeds for each calling process.
        """
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def _normal_perturb_dlxVector(self, sigma: float, out: dlx.la.Vector) -> None:
        """
        Add a normal perturbation to a dolfinx Vector.
        """
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        loc_random_numbers = np.random.default_rng(self.rng).normal(
            loc=0,
            scale=sigma,
            size=num_local_values,
        )
        out.array[:] += loc_random_numbers
        out.scatter_forward()

    def _normal_perturb_multivec(self, sigma: float, out: MultiVector) -> None:
        """
        Add a normal perturbation to a MultiVector.
        """
        num_local_values = out[0].getLocalSize()
        for i in range(out.nvec):
            loc_random_numbers = np.random.default_rng(self.rng).normal(
                loc=0,
                scale=sigma,
                size=num_local_values,
            )
            with out[i].localForm() as v_array:
                v_array[0:num_local_values] += loc_random_numbers

            out[i].ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.INSERT,  # type: ignore
                mode=petsc4py.PETSc.ScatterMode.FORWARD,  # type: ignore
            )

    def normal_perturb(self, sigma: float, out: Union[dlx.la.Vector, MultiVector]) -> None:
        """
        Add a normal perturbation to a dolfinx Vector or MultiVector object.
        """
        if hasattr(out, "nvec"):
            self._normal_perturb_multivec(sigma, out)  # type: ignore
        else:
            self._normal_perturb_dlxVector(sigma, out)

    def normal(self, sigma: float, out: Union[dlx.la.Vector, MultiVector]) -> None:
        """
        Sample from a normal distribution.
        """
        if hasattr(out, "scale"):
            out.scale(0.0)
        else:
            out.array[:] = 0.0

        self.normal_perturb(sigma, out)


_world_rank = MPI.COMM_WORLD.rank
_world_size = MPI.COMM_WORLD.size

parRandom = Random(_world_rank, _world_size)
