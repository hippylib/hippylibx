import numpy as np
from petsc4py import PETSc

from dolfinx import fem, geometry


def findPoints(mesh, x):
    # ------------------------------------------------------------
    # Find which cells contain the interpolation points
    # ------------------------------------------------------------
    tdim = mesh.topology.dim
    bb_tree = geometry.bb_tree(mesh, tdim)

    candidate_cells = geometry.compute_collisions_points(bb_tree, x)
    colliding_cells = geometry.compute_colliding_cells(mesh, candidate_cells, x)

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

    return local_points, local_cells, point_owner_rows


def dofLGmap(comm, index_map, bs):
    nlocal = index_map.size_local
    nghost = index_map.num_ghosts

    # Block dofs
    local_blocks = np.arange(nlocal + nghost, dtype=np.int32)

    global_blocks = index_map.local_to_global(local_blocks)

    # Expand block -> scalar numbering
    local_to_global = np.empty((nlocal + nghost) * bs, dtype=np.int32)

    for i, g in enumerate(global_blocks):
        for b in range(bs):
            local_to_global[i * bs + b] = g * bs + b

    lgmap = PETSc.LGMap().create(local_to_global, comm=comm)

    return lgmap


def valLGmap(comm, point_owner_rows, bs_val):
    row_l2g = np.empty(len(point_owner_rows) * bs_val, dtype=np.int32)

    for i, p in enumerate(point_owner_rows):
        for b in range(bs_val):
            row_l2g[i * bs_val + b] = p * bs_val + b

    row_lgmap = PETSc.LGMap().create(row_l2g, comm=comm)

    return row_lgmap


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
        raise ValueError("x must have shape (npoints, 3)")

    local_points, local_cells, point_owner_rows = findPoints(mesh, x)

    # ------------------------------------------------------------
    # Prepare PETSc matrix
    # ------------------------------------------------------------
    index_map = V.dofmap.index_map
    bs_dofs = V.dofmap.bs

    if len(V.element.value_shape) == 0:
        bs_val = 1
    else:
        bs_val = int(np.prod(V.element.value_shape))

    ndofs_global = index_map.size_global * bs_dofs
    ndofs_local = index_map.size_local * bs_dofs

    nrows_global = x.shape[0] * bs_val
    nrows_local = len(point_owner_rows) * bs_val

    P = PETSc.Mat().createAIJ(
        size=((nrows_local, nrows_global), (ndofs_local, ndofs_global)),
        comm=comm,
    )

    lgmap_dofs = dofLGmap(comm, index_map, bs_dofs)

    row_lgmap = valLGmap(comm, point_owner_rows, bs_val)

    P.setLGMap(row_lgmap, lgmap_dofs)

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
        x_ref = cmap.pull_back(xp.reshape(1, 3), cell_geometry)

        # Evaluate basis functions on reference cell
        basis = element.basix_element.tabulate(0, x_ref)[0, 0, :, :]

        # Cell dofs
        cell_dofs = V.dofmap.cell_dofs(cell)

        for b in range(bs_val):
            row = row_local * bs_val + b
            vals_b = basis[
                :, min([b, basis.shape[1] - 1])
            ]  # 0 for lagrangian elements, b for RT/ND

            P.setValuesLocal(
                [row],
                cell_dofs * bs_dofs + b,
                vals_b,
                addv=PETSc.InsertMode.ADD_VALUES,
            )

    P.assemblyBegin()
    P.assemblyEnd()

    return P
