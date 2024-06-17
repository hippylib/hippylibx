# Method 2 to create submesh from reduced labels.
import dolfinx as dlx
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

num_times_to_run = 1

time_compute_cell_center = 0.
time_compute_cells_to_keep = 0.
time_create_submesh = 0.

for _ in range(num_times_to_run):
    start_time = MPI.Wtime()
    reduced_labels = np.load('reduced_labels_factor_4_method_resize.npy')
    nx, ny, nz = reduced_labels.shape      
    num_cells_each_dimension = [nx, ny, nz]
    offset = np.array([-85,-85,-85])
    voxel_sizes = np.array([0.125, 0.125, 0.125])

    top_right_coordinates = [offset[i] + num_cells_each_dimension[i]*voxel_sizes[i] for i in range(3)] 
    msh = dlx.mesh.create_box(MPI.COMM_WORLD, [offset, top_right_coordinates], num_cells_each_dimension, dlx.mesh.CellType.hexahedron)
    geometry = msh.geometry.x
    connectivity = msh.topology.connectivity(msh.topology.dim, 0)
    cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_vertices = [connectivity.links(i) for i in range(cells)]
    cell_centers = np.array([geometry[vertices].mean(axis=0) for vertices in cell_vertices])
    end_time = MPI.Wtime()
    time_compute_cell_center += end_time - start_time

    start_time = MPI.Wtime()
    cells_to_keep = []
    for cell in range(cells):
        center = cell_centers[cell]
        i = int( (center[0] - offset[0])/voxel_sizes[0] )
        j = int( (center[1] - offset[1])/voxel_sizes[1] )
        k = int( (center[2] - offset[2])/voxel_sizes[2] )
        if(reduced_labels[i, j, k] != 0):
            cells_to_keep.append(cell)
    end_time = MPI.Wtime()
    time_compute_cells_to_keep += end_time - start_time
    
    cells_to_keep = np.array(cells_to_keep, dtype=np.int32)

    start_time = MPI.Wtime()
    submesh, cell_map, vertex_map, _ = dlx.mesh.create_submesh(msh, msh.topology.dim, cells_to_keep)
    end_time = MPI.Wtime()
    time_create_submesh += end_time - start_time

    with dlx.io.XDMFFile(submesh.comm, "submesh_3d_factor_4_method_resize.xdmf", "w") as xdmf:
        xdmf.write_mesh(submesh)

print(f'Average time to compute cell centers for {num_times_to_run} runs = {time_compute_cell_center/num_times_to_run} seconds')
print(f'Average time to compute list of cells to keep for {num_times_to_run} runs = {time_compute_cells_to_keep/num_times_to_run} seconds')
print(f'Average time to create submesh for {num_times_to_run} runs = {time_create_submesh/num_times_to_run} seconds')

