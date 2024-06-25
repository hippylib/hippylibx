# %% 
#script to interpolate a 3d numpy array of values over a mesh.
import dolfinx as dlx
from mpi4py import MPI
import numpy as np

fname = 'submesh_3d_factor_8_method_zoom.xdmf'
comm = MPI.COMM_WORLD
fid = dlx.io.XDMFFile(comm, fname, "r")
msh = fid.read_mesh(name="mesh")
optical_arr = np.load('reduced_opt_array_factor_8_method_zoom.npy') #(46, 103, 98)
offset_disp = np.load('offset_disp_factor_8_method_zoom.npy')

Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
m_true = dlx.fem.Function(Vh_m)

original_center = np.array([-85,-85,-85])
voxel_sizes = np.array([0.125, 0.125, 0.125])

#revsied centers:
offset = np.zeros(3)
for i in range(3):
    offset[i] = original_center[i] + voxel_sizes[i]*offset_disp[i]

def interpolate_function(x):

    i = np.clip(np.floor((x[0] - offset[0]) / voxel_sizes[0]).astype(int), 0, optical_arr.shape[0] - 1)
    j = np.clip(np.floor((x[1] - offset[1]) / voxel_sizes[1]).astype(int), 0, optical_arr.shape[1] - 1)
    k = np.clip(np.floor((x[2] - offset[2]) / voxel_sizes[2]).astype(int), 0, optical_arr.shape[2] - 1)
      
    return optical_arr[i, j, k]

# Interpolate values
m_true.interpolate(interpolate_function)

m_true.x.scatter_forward()

nproc = comm.size
with dlx.io.XDMFFile(
    msh.comm,
    "interpolated_optical_properties_np{0:d}.xdmf".format(nproc),
    "w",
) as file:
    file.write_mesh(msh)
    file.write_function(m_true,0)



    # %%
