import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, geometry
from ufl import TrialFunction


def pointwiseInterpolationMatrix(V: fem.FunctionSpace, x: np.ndarray) -> PETSc.Mat:
    """
    Build a PETSc sparse matrix P such that

        u_vals = P * u_vec

    interpolates a finite-element function u in the DOLFINx function
    space V at the spatial points x.

    Parameters
    ----------
    V
        DOLFINx function space.
    x
        Array of shape (npoints, gdim) containing spatial coordinates.

    Returns
    -------
    P : PETSc.Mat
        Parallel PETSc AIJ matrix of shape
            (npoints_global, V.dofmap.index_map.size_global * bs)

        where bs = V.dofmap.index_map_bs.

        Multiplying P by the global coefficient vector of a function in V
        gives the interpolated values at the requested points.
    """

    mesh = V.mesh
    comm = mesh.comm
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim

    x = np.asarray(x, dtype=np.float64)
    # DOLFINx geometry routines expect 3D coordinates
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] == 2:
        x3 = np.zeros((x.shape[0], 3), dtype=np.float64)
        x3[:, :2] = x
        x = x3
    elif x.shape[1] != 3:
        raise ValueError("Points must have shape (N,2) or (N,3)")

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must have shape (npoints, 3)")

    # ------------------------------------------------------------
    # Find which cells contain the interpolation points
    # ------------------------------------------------------------
    bb_tree = geometry.bb_tree(mesh, tdim)

    candidate_cells = geometry.compute_collisions_points(bb_tree, x)
    colliding_cells = geometry.compute_colliding_cells(
        mesh, candidate_cells, x
    )

    local_points = []
    local_cells = []
    point_owner_rows = []

    for i in range(x.shape[0]):
        cells = colliding_cells.links(i)

        if len(cells) > 0:
            local_points.append(x[i])
            local_cells.append(cells[0])
            point_owner_rows.append(i)

    local_points = np.array(local_points, dtype=np.float64)

    # ------------------------------------------------------------
    # Prepare PETSc matrix
    # ------------------------------------------------------------
    index_map = V.dofmap.index_map
    bs = V.dofmap.bs

    ndofs_global = index_map.size_global * bs
    nrows_global = x.shape[0]

    nrows_local = len(point_owner_rows)

    P = PETSc.Mat().createAIJ(
        size=((nrows_local, nrows_global), (ndofs_global, ndofs_global)),
        comm=comm,
    )

    # ------------------------------------------------------------
    # Tabulate basis functions at interpolation points
    # ------------------------------------------------------------
    element = V.element
    cmap = mesh.geometry.cmap
    x_g = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap

    for row_local, (row_global, cell, xp) in enumerate(
        zip(point_owner_rows, local_cells, local_points)
    ):
        # Geometry dofs for this cell
        geom_dofs = x_dofs[cell]
        cell_geometry = x_g[geom_dofs]

        # Pull physical point back to reference cell
        x_ref = cmap.pull_back(
            xp.reshape(1, gdim),
            cell_geometry
        )

        # Evaluate basis functions on reference cell
        basis = element.tabulate(0, x_ref)[0, 0, :, :]

        # Cell dofs
        cell_dofs = V.dofmap.cell_dofs(cell)

        # Scalar FE space
        if basis.shape[1] == 1:
            values = basis[:, 0]
            cols = cell_dofs

        # Vector/tensor FE space
        else:
            values = basis.reshape(-1)
            cols = np.repeat(cell_dofs, basis.shape[1])

        P.setValues(
            [row_local],
            cols,
            values,
            addv=PETSc.InsertMode.ADD_VALUES,
        )

    P.assemblyBegin()
    P.assemblyEnd()

    return P
