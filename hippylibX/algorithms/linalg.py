import dolfinx as dlx
import numpy as np
import mpi4py

# def inner( a: dlx.la.Vector, b: dlx.la.Vector):
#     '''
#     return the inner product of two dolfinx Vector objects
#     '''
#     out = np.zeros(1)
#     out[0] = np.inner(a.array, b.array)
#     comm = a.index_map.comm
#     comm.Allreduce(mpi4py.MPI.IN_PLACE, out, op=mpi4py.MPI.SUM)
#     return out[0]


def inner(a: dlx.la.Vector, b: dlx.la.Vector):
    '''
    return the inner product of two dolfinx Vector objects
    '''
    out = np.zeros(1)
    out[0] = np.inner(a.array, b.array)
    
    comm = a.index_map.comm
    comm.Allreduce(mpi4py.MPI.IN_PLACE, out, op=mpi4py.MPI.SUM)
    return out[0]


