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
    
    seed_val = rank

    rng = np.random.MT19937(seed_val)  
    
    glb_num_vals = d.vector.getSize()

    loc_num_vals_normal = int(glb_num_vals/nproc)
    

    if(rank == nproc - 1): #for last process
        loc_num_vals_last = glb_num_vals - ( nproc-1)*loc_num_vals_normal    
        loc_random_numbers = np.random.Generator(rng.jumped(1)).normal(scale=np.sqrt(noise_variance),size=loc_num_vals_last) 
        indices = np.arange(rank*loc_num_vals_normal, glb_num_vals, dtype=np.int32)
        loc_num_vals = loc_num_vals_last        

    else: #for all but last process
        loc_random_numbers = np.random.Generator(rng.jumped(1)).normal(scale=np.sqrt(noise_variance),size=loc_num_vals_normal)
        indices = np.arange( rank*loc_num_vals_normal, (rank + 1)*loc_num_vals_normal, dtype=np.int32)
        loc_num_vals = loc_num_vals_normal


    d.vector.setValues(indices,loc_random_numbers,addv=True)
    d.vector.assemblyBegin()
    d.vector.assemblyEnd()


    # v = petsc4py.PETSc.Viewer()
    # v(d.vector)
    


