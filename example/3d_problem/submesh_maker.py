#a separate file for creating the submesh so it may run in parallel: mpirun -n 4 ...:
#file to write the submesh to
from utils import submesh_maker
downsample_label_file = 'downsample_labels.h5' 
submesh_file = 'submesh_3d_problem.xdmf' 
submesh_maker(downsample_label_file, submesh_file)
