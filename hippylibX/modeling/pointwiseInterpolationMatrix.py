import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry


def findPoints(mesh, x):
    """
    Locate the points :code:`x` in the cells owned by this process.

    A point that lies on the interface between processes (or in a ghost cell)
    may be found by several processes; it is assigned to the lowest rank that
    owns a cell containing it, so that every point is handled exactly once.
    Points that are not found by any process (outside the mesh) are assigned
    to rank 0, with cell -1.

    Returns the local points, the cells containing them, and the indices of
    these points in :code:`x` (in increasing order).
    """
    comm = mesh.comm
    tdim = mesh.topology.dim
    num_owned_cells = mesh.topology.index_map(tdim).size_local
    bb_tree = geometry.bb_tree(mesh, tdim)

    candidate_cells = geometry.compute_collisions_points(bb_tree, x)
    colliding_cells = geometry.compute_colliding_cells(mesh, candidate_cells, x)

    npoints = x.shape[0]
    found_cell = np.full(npoints, -1, dtype=np.int32)
    for i in range(npoints):
        owned = colliding_cells.links(i)
        owned = owned[owned < num_owned_cells]
        if len(owned) > 0:
            found_cell[i] = owned[0]

    # Lowest rank that found the point wins
    candidate_rank = np.where(found_cell >= 0, comm.rank, comm.size).astype(np.int32)
    owner_rank = np.empty_like(candidate_rank)
    comm.Allreduce(candidate_rank, owner_rank, op=MPI.MIN)

    # Points outside the mesh go to rank 0
    owner_rank[owner_rank == comm.size] = 0

    point_ids = np.flatnonzero(owner_rank == comm.rank).astype(np.int32)
    local_cells = found_cell[point_ids]
    local_points = x[point_ids]

    return local_points, local_cells, point_ids


def dofLGmap(comm, index_map, bs):
    """Block local-to-global map of the (owned + ghost) dofs of a dofmap."""
    nlocal = index_map.size_local + index_map.num_ghosts
    global_blocks = index_map.local_to_global(np.arange(nlocal, dtype=np.int32))
    return PETSc.LGMap().create(
        global_blocks.astype(PETSc.IntType), bsize=bs, comm=comm
    )


def pointLGmap(comm, point_ids, bs_val):
    """
    Block local-to-global map of the rows of the interpolation matrix: local
    row block j corresponds to the point x[point_ids[j]], whose global row
    block is point_ids[j] (the same numbering as in serial).
    """
    return PETSc.LGMap().create(
        np.asarray(point_ids, dtype=PETSc.IntType), bsize=bs_val, comm=comm
    )


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
        The same array must be passed on every process.

    Returns
    -------
    P : PETSc.Mat
        Parallel PETSc AIJ matrix of shape
            (npoints * value_size, V.dofmap.index_map.size_global * bs)

        where bs = V.dofmap.index_map_bs.

        Row i * value_size + comp is component comp of the value at x[i],
        as in serial. Each process computes the rows of the points that lie
        in its owned cells (a point on an interface between processes goes to
        the lowest rank; points outside the mesh give zero rows and go to
        rank 0), and inserts them through the row local-to-global map
        ``P.getLGMap()[0]``, which maps local row block j to point_ids[j].
        Each process has as many local rows as points it computes; PETSc
        stores the rows in contiguous ranges of the global (x) ordering, so
        rows computed on one process may be stored on another.

        Multiplying P by the global coefficient vector of a function in V
        gives the interpolated values at the requested points.
    """

    mesh = V.mesh
    comm = mesh.comm

    family = V.element.basix_element.family.name
    if family in ["RT", "BDM", "N1E", "N2F"]:
        raise ValueError(f"Unsupported finite element family {family}")

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

    local_points, local_cells, point_ids = findPoints(mesh, x)

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
    nrows_local = len(point_ids) * bs_val

    ndofs_per_cell = V.dofmap.dof_layout.num_dofs

    P = PETSc.Mat().createAIJ(
        size=((nrows_local, nrows_global), (ndofs_local, ndofs_global)),
        nnz=(ndofs_per_cell, ndofs_per_cell),
        comm=comm,
    )
    P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    row_lgmap = pointLGmap(comm, point_ids, bs_val)
    col_lgmap = dofLGmap(comm, index_map, bs_dofs)
    P.setLGMap(row_lgmap, col_lgmap)

    # ------------------------------------------------------------
    # Tabulate basis functions at interpolation points
    # ------------------------------------------------------------
    element = V.element
    cmap = mesh.geometry.cmap
    x_g = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap

    for row_block, (cell, xp) in enumerate(zip(local_cells, local_points)):
        if cell < 0:  # point outside the mesh: zero row
            continue

        # Geometry dofs for this cell
        geom_dofs = x_dofs[cell]
        cell_geometry = x_g[geom_dofs]

        # Pull physical point back to reference cell
        x_ref = cmap.pull_back(xp.reshape(1, 3), cell_geometry)

        # Evaluate basis functions on reference cell and pull back
        basis = element.basix_element.tabulate(0, x_ref)[0, 0, :, :]

        # Cell dofs (local block indices)
        cell_dofs = V.dofmap.cell_dofs(cell).astype(PETSc.IntType)

        for comp in range(bs_val):
            row = row_block * bs_val + comp

            if bs_dofs == bs_val:  # Lagrangian elements
                cols = cell_dofs * bs_dofs + comp
                vals_b = basis[:, 0]
            elif bs_dofs == 1:  # RT/ND
                cols = cell_dofs
                vals_b = basis[:, comp]
            else:
                raise ValueError(f"Unsupported block size bs={bs_dofs}")

            P.setValuesLocal(
                [row],
                cols,
                vals_b,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    P.assemblyBegin()
    P.assemblyEnd()

    return P
