# %% 
#script to interpolate a 3d numpy array of values over a mesh.
import dolfinx as dlx
from mpi4py import MPI

fname = 'submesh_3d_factor_8_method_zoom.xdmf'
comm = MPI.COMM_WORLD
fid = dlx.io.XDMFFile(comm, fname, "r")
msh = fid.read_mesh(name="mesh")

#optical properties array to interpolate over the above mesh
import numpy as np
optical_arr = np.load('reduced_opt_array_factor_8_method_zoom.npy') #(46, 103, 98)
offset_disp = np.load('offset_disp_factor_8_method_zoom.npy')
# %%
Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
m_true = dlx.fem.Function(Vh_m)
geometry = msh.geometry.x
connectivity = msh.topology.connectivity(msh.topology.dim, 0)
cells = msh.topology.index_map(msh.topology.dim).size_local
cell_vertices = [connectivity.links(i) for i in range(cells)]
cell_centers = np.array([geometry[vertices].mean(axis=0) for vertices in cell_vertices])

m_true_values = np.zeros(cells)

original_center = np.array([-85,-85,-85])
voxel_sizes = np.array([0.125, 0.125, 0.125])

offset = np.zeros(3)
for i in range(3):
    offset[i] = original_center[i] + voxel_sizes[i]*offset_disp[i]

#need to know (i, j, k) of cells in submesh in the tight matrix computed by removing slices of all 0s
counter = 0
for cell in range(cells):
    center = cell_centers[cell]    
    i = int( (center[0] - original_center[0])/voxel_sizes[0] )
    j = int( (center[1] - original_center[1])/voxel_sizes[1] )
    k = int( (center[2] - original_center[2])/voxel_sizes[2] )
    m_true_values[cell] = optical_arr[i, j, k]

with m_true.vector.localForm() as local_m_true:
    local_m_true.setArray(m_true_values)
    

# print(len(m_true_values))
# print(len(m_true.x.array))
# print(len(geometry))
# m_true.x.scatter_forward()


    # counter += 1
    # if(counter == 10):
    #     break

# print(counter)
# m_true.interpolate(
#     lambda x: np.log(0.01)
#     + 3.0 * (((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0)) < 1.0)
# )

# m_true.x.scatter_forward()

# m_true = m_true.x


# %%
# nproc = comm.size
# with dlx.io.XDMFFile(
#     msh.comm,
#     "interpolated_function_np{0:d}.xdmf".format(nproc),
#     "w",
# ) as file:
#     file.write_mesh(msh)
#     file.write_function(m_true,0)

# # %%
# mesh_coords = msh.geometry.x
# print(mesh_coords)