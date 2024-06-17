# reduced labels using skimage.transform.resize

import timeit
import h5py
from typing import Tuple
import numpy as np
from skimage.transform import resize
import time

def label_maker(label_array: np.array, f:float )-> Tuple[np.array, np.array, np.array, np.array]:
    '''
    Function takes in the label array of dimension (680,1360,1360) and removes all 
    slices with all 0s (background) and downsamples by a factor 'f' and returns 
    x,y, and z ranges of the retained planes and the downsampled numpy array of label values. 
    '''
    i_range = np.max(label_array, axis=(1, 2)) > 0
    j_range = np.max(label_array, axis=(0, 2)) > 0
    k_range = np.max(label_array, axis=(0, 1)) > 0
    tight_label = label_array[i_range, :, :]
    tight_label = tight_label[:, j_range, :]
    tight_label = tight_label[:, :, k_range]
    revised_size = tuple( [tight_label.shape[0]//f, tight_label.shape[1]//f, tight_label.shape[2]//f ] )
    reduced_labels = resize(tight_label, revised_size, order = 0)
  
    return [i_range, j_range, k_range, reduced_labels]

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, result


file_path = 'A40210923l_label.mat'
with h5py.File(file_path, 'r') as f:
    data = f['label'][:].astype(np.int32)    

factor = 8
execution_time, result = time_function(label_maker, data, factor)
print(execution_time)
i_range, j_range, k_range, reduced_labels = result
np.save('i_range_factor_8_method_resize',i_range)
np.save('j_range_factor_8_method_resize',j_range)
np.save('k_range_factor_8_method_resize',k_range)
np.save('reduced_labels_factor_8_method_resize',reduced_labels)

print(reduced_labels.shape)
