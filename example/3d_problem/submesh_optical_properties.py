#script to create submesh and array of optical properties that can be interpolated over the submesh
# Input: label file(.mat), optical file (.mat), factor of reduction,
# Output: submesh, reduced array of optical properties

from utils import downsample_labels, submesh_maker, interpolate_optical_properties

factor = 4
labels_file =  'A40210923l_label.mat'
optical_props_file = 'A40210923l_opt.mat'

downsample_label_file = 'downsample_labels.h5'

downsample_labels(factor, labels_file, downsample_label_file)

submesh_file = 'submesh_3d_problem.xdmf'
submesh_maker(downsample_label_file, submesh_file)

downsample_optical_file = 'downsample_optical_properties.h5'
interpolate_optical_properties(downsample_label_file, optical_props_file, downsample_optical_file)







