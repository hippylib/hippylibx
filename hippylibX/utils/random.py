import numpy as np
import petsc4py
from mpi4py import MPI
import dolfinx as dlx
from ..algorithms.multivector import MultiVector
from typing import Union

class Random:
    def __init__(self, rank : int, nproc : int, seed=1) -> None:
        seed_sequence = np.random.SeedSequence(seed)
        self.child_seeds = seed_sequence.spawn(nproc)
        self.rank = rank
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def replay(self) -> None:
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def _normal_perturb_dlxVector(self, sigma : float, out : dlx.la.Vector) -> None:
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        loc_random_numbers = np.random.default_rng(self.rng).normal(
            loc=0, scale=sigma, size=num_local_values
        )
        out.array[:] += loc_random_numbers
        out.scatter_forward()

    def _normal_perturb_multivec(self, sigma : float, out : MultiVector) -> None:
        num_local_values = out[0].getLocalSize()
        for i in range(out.nvec):
            loc_random_numbers = np.random.default_rng(self.rng).normal(
                loc=0, scale=sigma, size=num_local_values
            )
            with out[i].localForm() as v_array:
                v_array[0:num_local_values] += loc_random_numbers

            out[i].ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.INSERT,
                mode=petsc4py.PETSc.ScatterMode.FORWARD,
            )

    def normal_perturb(self, sigma : float, out : Union[dlx.la.Vector, MultiVector] ) -> None:
        if hasattr(out, "nvec"):
            self._normal_perturb_multivec(sigma, out)
        else:
            self._normal_perturb_dlxVector(sigma, out)

    def normal(self, sigma : float, out : Union[dlx.la.Vector, MultiVector] ) -> None:
        if hasattr(out, "scale"):
            out.scale(0.0)
        else:
            out.array[:] = 0.0

        self.normal_perturb(sigma, out)


_world_rank = MPI.COMM_WORLD.rank
_world_size = MPI.COMM_WORLD.size

parRandom = Random(_world_rank, _world_size)
