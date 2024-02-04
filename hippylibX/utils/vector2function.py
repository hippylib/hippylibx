import dolfinx as dlx
import petsc4py

def create_petsc_vector_wrap(x, comm):
    #x -> dolfinx.la.Vector object
    u_map = x.map
    ghosts = u_map.ghosts.astype(petsc4py.PETSc.IntType)
    bs = x.bs
    u_size = (u_map.size_local *bs, u_map.size_global*bs)

    return petsc4py.PETSc.Vec().createGhostWithArray(ghosts, x.array, size=u_size, bsize=bs, comm = comm)
    

def vector2Function(vec,Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """
    # fun = dlx.fem.Function(Vh,**kwargs)
    # fun.vector.setArray(x.getArray())
    # # Make sure fun is ready to be used for assembly or plotting
    # fun.x.scatter_forward()
    # return fun

    #x is petsc4pyVec    
    # fun = dlx.fem.Function(Vh,**kwargs)
    # # fun.vector.axpy(1., x)

    # fun.vector.setArray(x.getArray())
    # fun.x.scatter_forward()
    # return fun

    #x is dlx.la.Vector object
    fun = dlx.fem.Function(Vh,**kwargs)
    # print(len(fun.x.array),":",len(vec.array))
    fun.x.array[:] = vec.array[:]
    return fun
    
    # fun = dlx.fem.Function(Vh,**kwargs)
    # fun_vec = fun.vector
    # fun_vec.axpy(1., x)
    # fun.x.scatter_forward()
    # return fun
