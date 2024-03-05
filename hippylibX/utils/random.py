import numpy as np
import dolfinx as dlx
import petsc4py

class parRandom():
    def __init__(self, comm):
        #All the seeding of the random number generator should 
        #happen once for all in the init method.
        rank = comm.rank
        nproc = comm.size

        master_seed = 123
        seed_sequence = np.random.SeedSequence(master_seed)

        child_seeds = seed_sequence.spawn(nproc)
        self.rng = np.random.MT19937(child_seeds[rank])

    def normal(self,sigma, out):
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        loc_random_numbers = np.random.default_rng(self.rng).normal(loc=0,scale= sigma,size=num_local_values)
        
        out.array[:] = loc_random_numbers
        dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)

    def normal_perturb(self,sigma, out):
        num_local_values = out.index_map.size_local + out.index_map.num_ghosts
        loc_random_numbers = np.random.default_rng(self.rng).normal(loc=0,scale= sigma,size=num_local_values)
        out.array[:] += loc_random_numbers
        dlx.la.create_petsc_vector_wrap(out).ghostUpdate(petsc4py.PETSc.InsertMode.INSERT,petsc4py.PETSc.ScatterMode.FORWARD)


