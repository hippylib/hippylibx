import dolfinx as dlx
import petsc4py


def vector2Function(vec,Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """

    #x is dlx.la.Vector object
    fun = dlx.fem.Function(Vh,**kwargs)
    # print(len(fun.x.array),":",len(vec.array))
    fun.x.array[:] = vec.array[:]
    fun.x.scatter_forward()
    return fun
