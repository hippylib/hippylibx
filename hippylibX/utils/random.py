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

    def normal(self, sigma, out):
        if(hasattr(out,"nvec")):
            num_local_values = out[0].getLocalSize()
        else:
            num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        
        loc_random_numbers = np.random.default_rng(self.rng).normal(
            loc=0, scale=sigma, size=num_local_values
        )
        if(hasattr(out,"nvec")): #for multivec
            for i in range(out.nvec):
                out[i].setArray(loc_random_numbers)                
        else:
            out.array[:] = loc_random_numbers
            dlx.la.create_petsc_vector_wrap(out).ghostUpdate(
                petsc4py.PETSc.InsertMode.INSERT, petsc4py.PETSc.ScatterMode.FORWARD
            )

    def normal_perturb(self, sigma, out):

        if(hasattr(out,"nvec")):
            num_local_values = out[0].getLocalSize()
        else:
            num_local_values = out.index_map.size_local + out.index_map.num_ghosts
    
        loc_random_numbers = np.random.default_rng(self.rng).normal(
            loc=0, scale=sigma, size=num_local_values
        )
        if(hasattr(out,"nvec")): #for multivec
            for i in range(out.nvec):
                # out[i] += loc_random_numbers
                out[i].axpy(1., loc_random_numbers) #FIXME: add values to petsc Vec               
        else:
            out.array[:] += loc_random_numbers
            dlx.la.create_petsc_vector_wrap(out).ghostUpdate(
                petsc4py.PETSc.InsertMode.INSERT, petsc4py.PETSc.ScatterMode.FORWARD
            )


_world_rank = MPI.COMM_WORLD.rank
_world_size = MPI.COMM_WORLD.size

parRandom = Random(_world_rank, _world_size)
