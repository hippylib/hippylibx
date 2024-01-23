import dolfinx as dlx

def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """
    # fun = dlx.fem.Function(Vh,**kwargs)
    # fun.vector.setArray(x.getArray())
    # # Make sure fun is ready to be used for assembly or plotting
    # fun.x.scatter_forward()
    # return fun
    
    fun = dlx.fem.Function(Vh,**kwargs)
    # fun.vector.axpy(1., x)
    fun.vector.setArray(x.getArray())
    fun.x.scatter_forward()
    return fun

    # fun = dlx.fem.Function(Vh,**kwargs)
    # fun_vec = fun.vector
    # fun_vec.axpy(1., x)
    # fun.x.scatter_forward()
    # return fun
