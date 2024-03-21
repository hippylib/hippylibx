import numpy as np
import dolfinx as dlx
import petsc4py
from mpi4py import MPI

class Random():
    def __init__(self, rank, nproc,seed = 1):
        seed_sequence = np.random.SeedSequence(seed)
        self.child_seeds = seed_sequence.spawn(nproc)
        self.rank = rank

    def replay(self):
        self.rng = np.random.MT19937(self.child_seeds[self.rank])
        
    def normal(self,sigma, out):
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        self.replay()
        loc_random_numbers = np.random.default_rng(self.rng).normal(loc=0,scale= sigma,size=num_local_values)
        
        out.array[:] = loc_random_numbers
        dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

    def normal_perturb(self,sigma, out):
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        self.replay()
        loc_random_numbers = np.random.default_rng(self.rng).normal(loc=0,scale= sigma,size=num_local_values)
        out.array[:] += loc_random_numbers
        dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)


_world_rank = MPI.COMM_WORLD.rank
_world_size = MPI.COMM_WORLD.size

parRandom = Random(_world_rank,_world_size)