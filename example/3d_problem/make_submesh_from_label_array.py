import dolfinx as dlx
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

num_times_to_run = 5
total_time = 0.
for _ in range(num_times_to_run):
    start_time = MPI.Wtime()

    i_range = np.load('i_range_factor_8.npy')
    j_range = np.load('j_range_factor_8.npy')
    k_range = np.load('k_range_factor_8.npy')

    reduced_labels = np.load('reduced_labels_factor_8_colab.npy')
        
    nx, ny, nz = reduced_labels.shape

    msh = dlx.mesh.create_unit_cube(MPI.COMM_WORLD, nx, ny, nz, dlx.mesh.CellType.hexahedron)

    geometry = msh.geometry.x
    connectivity = msh.topology.connectivity(msh.topology.dim, 0)
    cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_vertices = [connectivity.links(i) for i in range(cells)]
    cell_centers = np.array([geometry[vertices].mean(axis=0) for vertices in cell_vertices])

    #all cells associated with label = 0 are excluded.
    cells_to_keep = []
    for cell in range(cells):
        center = cell_centers[cell]
        i = int(center[0] * nx)
        j = int(center[1] * ny)
        k = int(center[2] * nz) 
        if(reduced_labels[i, j, k] != 0):
            cells_to_keep.append(cell)

    cells_to_keep = np.array(cells_to_keep, dtype=np.int32)
    # print(labels, cells_to_keep)
    # #create submesh:
    submesh, cell_map, vertex_map, _ = dlx.mesh.create_submesh(msh, msh.topology.dim, cells_to_keep)
    end_time = MPI.Wtime()
    total_time += end_time - start_time

print(total_time/num_times_to_run)

with dlx.io.XDMFFile(submesh.comm, "submesh_3d_factor_8.xdmf", "w") as xdmf:
    xdmf.write_mesh(submesh)
