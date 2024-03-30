import dolfinx as dlx


def vector2Function(vec, Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """

    fun = dlx.fem.Function(Vh, **kwargs)
    fun.x.array[:] = vec.array[:]
    fun.x.scatter_forward()
    return fun


def updateFromVector(fun: dlx.fem.Function, vec: dlx.la.Vector):
    fun.x.array[:] = vec.array[:]
    fun.x.scatter_forward()
