import numpy as np

def parRandom(comm,noise_variance,d):
    """
    comm -> communicator
    noise_variance -> the variance of the normal distribution used to perturb the vector associated with the function d
    d -> the vector associated with this function will be perturbed with the noise
    """

    rank  = comm.rank
    nproc = comm.size

    glb_num_vals = d.vector.getSize()
    seeds = np.random.randint(0,2**32,size=nproc)
    loc_seed = comm.scatter(seeds,root=0)
    rng = np.random.default_rng(loc_seed)

    loc_num_vals_normal = int(glb_num_vals/nproc)
    
    if(rank == nproc - 1): #for last process
        loc_num_vals_last = glb_num_vals - ( nproc-1)*loc_num_vals_normal    
        loc_random_numbers = rng.normal(loc=0,scale=np.sqrt(noise_variance),size=loc_num_vals_last)
        indices = np.arange(rank*loc_num_vals_normal, glb_num_vals, dtype=np.int32)
    
    else: #for all but last process
        loc_random_numbers = rng.normal(loc=0,scale=np.sqrt(noise_variance),size=loc_num_vals_normal)
        indices = np.arange( rank*loc_num_vals_normal, (rank + 1)*loc_num_vals_normal, dtype=np.int32)
    
    d.vector.setValues(indices,loc_random_numbers,addv=True)
    d.vector.assemblyBegin()
    d.vector.assemblyEnd()
    
    #All values in the vector seem to have been perturbed
    # v = petsc4py.PETSc.Viewer()
    # v(d.vector)