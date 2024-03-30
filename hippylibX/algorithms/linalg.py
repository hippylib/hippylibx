import dolfinx as dlx


def inner(x: dlx.la.Vector, y: dlx.la.Vector):
    return dlx.cpp.la.inner_product(x._cpp_object, y._cpp_object)
