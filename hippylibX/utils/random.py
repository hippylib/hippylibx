import numpy as np
import dolfinx as dlx
import petsc4py
from mpi4py import MPI


class Random:
    def __init__(self, rank, nproc, seed=1):
        seed_sequence = np.random.SeedSequence(seed)
        self.child_seeds = seed_sequence.spawn(nproc)
        self.rank = rank
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def replay(self):
        self.rng = np.random.MT19937(self.child_seeds[self.rank])

    def _normal_perturb_vec(self, sigma, out):
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        loc_random_numbers = np.random.default_rng(self.rng).normal(
            loc=0, scale=sigma, size=num_local_values
        )
        out.array[:] += loc_random_numbers
        out.scatter_forward()

    def _normal_perturb_multivec(self, sigma, out):
        num_local_values = out[0].getLocalSize()
        for i in range(out.nvec):
            loc_random_numbers = np.random.default_rng(self.rng).normal(
                loc=0, scale=sigma, size=num_local_values
            )
            out[i].setValues(
                np.arange(
                    out[i].getOwnershipRange()[0],
                    out[i].getOwnershipRange()[1],
                    dtype=np.int32,
                ),
                loc_random_numbers,
                addv=petsc4py.PETSc.InsertMode.ADD,
            )
            out[i].assemble()

    def normal_perturb(self, sigma, out):
        if hasattr(out, "nvec"):
            self._normal_perturb_multivec(sigma, out)
        else:
            self._normal_perturb_vec(sigma, out)

    def normal(self, sigma, out):
        if hasattr(out, "scale"):
            out.scale(0.0)
        else:
            out.array[:] = 0.0

        self.normal_perturb(sigma, out)


_world_rank = MPI.COMM_WORLD.rank
_world_size = MPI.COMM_WORLD.size

parRandom = Random(_world_rank, _world_size)
