import h5py
import numpy as np
# from scipy.ndimage import zoom
import skimage.measure
import dolfinx as dlx
from mpi4py import MPI

def downsample_labels(factor:float, labels_file: str, downsampled_label_file: str) -> None:   
    '''
    Input:
        factor: factor of reduction
        labels_file: .mat file of label values
        downsampled_label_file: .h5 file to write out the new downsampled label array and its 
                                attributes (origin, voxel size, offsets).
    '''
    with h5py.File(labels_file, 'r') as f:
        data = f['label'][:].astype(np.int32)    
        origin = f['origin'][:]
        voxel_size = f['voxel_size'][:]

    #read in original center and voxel sizes
    # scale = 1/factor
    i_range = np.max(data, axis=(1, 2)) > 0
    j_range = np.max(data, axis=(0, 2)) > 0
    k_range = np.max(data, axis=(0, 1)) > 0
    tight_label = data[i_range, :, :]
    tight_label = tight_label[:, j_range, :]
    tight_label = tight_label[:, :, k_range]    
    # reduced_labels = zoom(tight_label,(scale ,scale,scale), order=0)
    reduced_labels = skimage.measure.block_reduce(tight_label,(factor,factor,factor),np.max)

    # first non-zero value in i, j, k_range:
    i_offset = np.where(i_range == True)[0][0]
    j_offset = np.where(j_range == True)[0][0]
    k_offset = np.where(k_range == True)[0][0]
    new_origin = np.array([origin.flatten()[0] + i_offset*voxel_size,
                        origin.flatten()[1] + j_offset*voxel_size,
                        origin.flatten()[2] + k_offset*voxel_size,
                        ])
    # print(origin.flatten(), new_origin.flatten())
    with h5py.File(f'{downsampled_label_file}', 'w') as h5f:
        h5f.create_dataset('factor', data=np.array([factor]))
        h5f.create_dataset('new_origin', data=new_origin.flatten())
        h5f.create_dataset('new_voxel_size', data=np.array([voxel_size*factor]))
        h5f.create_dataset('i_range', data=i_range)
        h5f.create_dataset('j_range', data=j_range)
        h5f.create_dataset('k_range', data=k_range)
        h5f.create_dataset('reduced_labels', data=reduced_labels)


def submesh_maker(downsample_label_file: str, submesh_file) -> None:
    '''
    Function to create a submesh using the downsampled label array
    Input: 
        downsample_label_file: path to downsampled label array constructed using function "downsample_labels"
        submesh_file: Name of submesh file to write submesh to 
    '''
    
    start_time =  MPI.Wtime()
    with h5py.File(f'{downsample_label_file}', 'r') as f:
        factor = f['factor'][:][0]
        origin = f['new_origin'][:]
        voxel_size = f['new_voxel_size'][:].flatten()[0]
        reduced_labels = f['reduced_labels'][:]

    nx, ny, nz = reduced_labels.shape      
    num_cells_each_dimension = [nx, ny, nz]

    top_right_coordinates = [origin[i] + num_cells_each_dimension[i]*voxel_size for i in range(3)] 
    msh = dlx.mesh.create_box(MPI.COMM_WORLD, [origin, top_right_coordinates], num_cells_each_dimension, dlx.mesh.CellType.hexahedron)
    end_time = MPI.Wtime()
    print(f'Time to create brick mesh = {end_time - start_time} seconds.')


    # msh.topology.create_connectivity(msh.topology.dim, 0)
    # msh.topology.create_connectivity(3, 2)
    # msh.topology.create_connectivity(3, 1)
    # msh.topology.create_connectivity(3, 3)

    start_time = MPI.Wtime()
    geometry = msh.geometry.x
    connectivity = msh.topology.connectivity(msh.topology.dim, 0)
    cell_indices = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)
    cell_centers = dlx.mesh.compute_midpoints(msh, msh.topology.dim, cell_indices)
    end_time = MPI.Wtime()
    print(f'Time to compute cell centers = {end_time - start_time} seconds.')

    start_time = MPI.Wtime()
    ijk_indices = np.floor((cell_centers - origin)/voxel_size).astype(int)
    cells_to_keep = cell_indices[np.where(reduced_labels[ijk_indices[:, 0], ijk_indices[:, 1], ijk_indices[:, 2]] != 0)[0]]
    cells_to_keep = np.array(cells_to_keep, dtype=np.int32)
    end_time = MPI.Wtime()
    print(f'Time to compute cells to keep = {end_time - start_time} seconds.')

    start_time = MPI.Wtime()
    submesh, _, _, _ = dlx.mesh.create_submesh(msh, msh.topology.dim, cells_to_keep)
    end_time = MPI.Wtime()
    print(f'Time to create submesh = {end_time - start_time} seconds.')


    with dlx.io.XDMFFile(submesh.comm, f"{submesh_file}", "w") as xdmf:
        xdmf.write_mesh(submesh)


def downsample_optical_properties(downsampled_label_file, opt_file, downsampled_optical_file) -> None:
  '''
  Function to downsample the optical properties array using the downsampled label array created using function
  "donwsample_labels".
  Input:
    downsampled_label_file: downsampled label array file
    opt_file: (.mat) file of optical properties
    downsampled_optical_file: (.h5) file to write the downsampled optical properties to
  '''
  
  with h5py.File(opt_file, 'r') as f:
    opt_data = f['mu_a_w757'][:]    

  with h5py.File(downsampled_label_file, 'r') as f:
    factor = f['factor'][:][0]
    i_range = f['i_range'][:]
    j_range = f['j_range'][:]
    k_range = f['k_range'][:]
  
  tight_opt_array = opt_data[i_range,:,:]
  tight_opt_array = tight_opt_array[:,j_range,:]
  tight_opt_array = tight_opt_array[:,:,k_range]

  # scale = 1/factor
  # reduced_opt_array  = zoom(tight_opt_array,(scale ,scale,scale), order=0)
  reduced_opt_array = skimage.measure.block_reduce(tight_opt_array,(factor,factor,factor),np.max)

  with h5py.File(f'{downsampled_optical_file}', 'w') as h5f:
    h5f.create_dataset('reduced_opt_array', data=reduced_opt_array)
  