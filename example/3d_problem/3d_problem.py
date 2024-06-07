##########working example 1
import numpy as np
import dolfinx as dlx
from dolfinx.mesh import create_unit_square, meshtags
from mpi4py import MPI
import dolfinx.io
import sys
import os

sys.path.append(os.environ.get("HIPPYLIBX_BASE_DIR", "../../"))
import hippylibX as hpx


comm = MPI.COMM_WORLD
nproc = comm.size

num_times_run = 1
total_time = 0.
for _ in range(num_times_run):
    start_time = MPI.Wtime()
    mesh = dlx.mesh.create_unit_cube(comm, 3, 3, 2, dolfinx.mesh.CellType.tetrahedron)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    labels = np.zeros(num_cells, dtype=np.int32)

    cell_midpoints = np.zeros((num_cells, 3))
    for i in range(num_cells):
        cell = mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(i)]
        cell_midpoints[i] = cell.mean(axis=0)[:]

    # 1. Labels chosen at specific location
    labels[cell_midpoints[:, 1] > 0.5] = 1 #top half: 1, bottom: 0
    # 2. labels chosen at random
    labels = np.random.randint(0,2, num_cells).astype(np.int32)

    mesh_tags = meshtags(mesh, mesh.topology.dim, np.arange(num_cells), labels)

    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    
# Print the elapsed time``
print(f"Average Elapsed time: {total_time/num_times_run:.6f} seconds")

with dlx.io.XDMFFile(mesh.comm, "submesh_2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(mesh_tags, mesh.geometry)
