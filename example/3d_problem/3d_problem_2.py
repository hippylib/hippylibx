# Goal: Take numpy array of indices in "array" and use a lambda function to interpolate it over the mesh.
import numpy as np
from mpi4py import MPI
import dolfinx as dlx

nx, ny = 10, 10
msh = dlx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [nx, ny])
V = dlx.fem.functionspace(msh, ("Lagrange", 1))

# CASE 1: randomly chosen points
# array = np.random.randint(0, 2, (nx + 1, ny + 1)) #random chosen indices, extend it to user-defined spatial locations

# CASE 2: for top half of mesh
array = np.zeros((nx + 1, ny + 1))
mid_y = ny // 2
array[:, mid_y:] = 1  # Labels for top half

def np_array_2_function(x, array):
    x_indices = (x[0] * nx).astype(int)
    y_indices = (x[1] * ny).astype(int)
    return array[x_indices, y_indices]

u = dlx.fem.Function(V)

u.interpolate(lambda x: np_array_2_function(x, array))

def cell_marker_function(x, array):
    values = np_array_2_function(x, array)
    return values >= 0.5

all_cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local)
marked_cells = dlx.mesh.locate_entities(msh, msh.topology.dim, lambda x: cell_marker_function(x, array))

submesh, cell_map, vertex_map, _  = dlx.mesh.create_submesh(msh, msh.topology.dim, marked_cells)

with dlx.io.XDMFFile(msh.comm, "submesh_3.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u)

with dlx.io.XDMFFile(submesh.comm, "submesh_top_half.xdmf", "w") as xdmf:
    xdmf.write_mesh(submesh)

#*****************************************************

# with dlx.io.XDMFFile(msh.comm, "submesh_3.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     xdmf.write_function(u)
    



# #CASE 3:  circle - not working
# #for circle within mesh - doesn't work! creates jagged domain instead. fix this!
# array = np.zeros((nx+1, ny+1))
# #idea: set 1s at (x,y) that are in the circular domain
# x_center, y_center = 0.5, 0.5
# rad = 0.3
# for i in range(nx + 1):
#     for j in range(ny + 1):
#         x = i / nx
#         y = j / ny
#         distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
#         if distance <= rad:
#             array[i, j] = 1

# def np_array_2_function(x, array):
#     x_indices = (x[0] * nx).astype(int)
#     y_indices = (x[1] * ny).astype(int)
#     return array[x_indices, y_indices]

# u = dlx.fem.Function(V)
# u.interpolate(lambda x: np_array_2_function(x, array))

# with dlx.io.XDMFFile(msh.comm, "submesh_4.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     xdmf.write_function(u)



# #CASE 4: circle , not  working:
# x_center, y_center = 0.5, 0.5
# rad = 0.3
# # def circle_domain(x):
# #     return np.where(np.sqrt((x[0] - x_center)**2 + (x[1] - y_center)**2) <= rad, 1.0, 0.0)
# comm = MPI.COMM_WORLD
# msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
# Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
# u = dlx.fem.Function(Vh_m)
# u.interpolate(lambda x: np.log(2 + 7 * (((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.5 > 0.2)))
# u.x.scatter_forward()

# with dlx.io.XDMFFile(msh.comm, "submesh_5.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     xdmf.write_function(u)


# msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
# Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
# m_true = dlx.fem.Function(Vh_m)
# m_true.interpolate(
#         lambda x: np.log(2 + 7 * (((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.5 > 0.2))
#     )
# m_true.x.scatter_forward()
# with dlx.io.XDMFFile(msh.comm, "submesh_6.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     xdmf.write_function(m_true)

