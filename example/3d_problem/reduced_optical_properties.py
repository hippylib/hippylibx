import numpy as np
opt_arr = np.load('optical_property_mu_a_w757.npy')

i_range_factor_8_method_zoom = np.load('i_range_factor_8_method_zoom.npy')
j_range_factor_8_method_zoom = np.load('j_range_factor_8_method_zoom.npy')
k_range_factor_8_method_zoom = np.load('k_range_factor_8_method_zoom.npy')

i_range_factor_4_method_zoom = np.load('i_range_factor_4_method_zoom.npy')
j_range_factor_4_method_zoom = np.load('j_range_factor_4_method_zoom.npy')
k_range_factor_4_method_zoom = np.load('k_range_factor_4_method_zoom.npy')

i_range_factor_8_method_resize = np.load('i_range_factor_8_method_resize.npy')
j_range_factor_8_method_resize = np.load('j_range_factor_8_method_resize.npy')
k_range_factor_8_method_resize = np.load('k_range_factor_8_method_resize.npy')

i_range_factor_4_method_resize = np.load('i_range_factor_4_method_resize.npy')
j_range_factor_4_method_resize = np.load('j_range_factor_4_method_resize.npy')
k_range_factor_4_method_resize = np.load('k_range_factor_4_method_resize.npy')

reduced_labels_factor_8_method_zoom = np.load('reduced_labels_factor_8_method_zoom.npy')
reduced_labels_factor_4_method_zoom = np.load('reduced_labels_factor_4_method_zoom.npy')

reduced_labels_factor_8_method_resize = np.load('reduced_labels_factor_8_method_resize.npy')
reduced_labels_factor_4_method_resize = np.load('reduced_labels_factor_4_method_resize.npy')

tight_opt_array_factor_8_method_zoom = opt_arr[i_range_factor_8_method_zoom,:,:]
tight_opt_array_factor_8_method_zoom = tight_opt_array_factor_8_method_zoom[:,j_range_factor_8_method_zoom,:]
tight_opt_array_factor_8_method_zoom = tight_opt_array_factor_8_method_zoom[:,:,k_range_factor_8_method_zoom]

tight_opt_array_factor_4_method_zoom = opt_arr[i_range_factor_4_method_zoom,:,:]
tight_opt_array_factor_4_method_zoom = tight_opt_array_factor_4_method_zoom[:,j_range_factor_4_method_zoom,:]
tight_opt_array_factor_4_method_zoom = tight_opt_array_factor_4_method_zoom[:,:,k_range_factor_4_method_zoom]

tight_opt_array_factor_8_method_resize = opt_arr[i_range_factor_8_method_resize,:,:]
tight_opt_array_factor_8_method_resize = tight_opt_array_factor_8_method_resize[:,j_range_factor_8_method_resize,:]
tight_opt_array_factor_8_method_resize = tight_opt_array_factor_8_method_resize[:,:,k_range_factor_8_method_resize]

tight_opt_array_factor_4_method_resize = opt_arr[i_range_factor_4_method_resize,:,:]
tight_opt_array_factor_4_method_resize = tight_opt_array_factor_4_method_resize[:,j_range_factor_4_method_resize,:]
tight_opt_array_factor_4_method_resize = tight_opt_array_factor_4_method_resize[:,:,k_range_factor_4_method_resize]


from scipy.ndimage import zoom
f = 8
scale = 1/f
reduced_opt_array_factor_8_method_zoom  = zoom(tight_opt_array_factor_8_method_zoom,(scale ,scale,scale), order=0)
reduced_opt_array_factor_8_method_zoom  = zoom(tight_opt_array_factor_8_method_zoom,(scale ,scale,scale), order=0)
f = 4
scale = 1/f
reduced_opt_array_factor_4_method_zoom  = zoom(tight_opt_array_factor_4_method_zoom,(scale ,scale,scale), order=0)
reduced_opt_array_factor_4_method_zoom  = zoom(tight_opt_array_factor_4_method_zoom,(scale ,scale,scale), order=0)

from skimage.transform import resize
f = 8
revised_size = tuple( [tight_opt_array_factor_8_method_resize.shape[0]//f, tight_opt_array_factor_8_method_resize.shape[1]//f, tight_opt_array_factor_8_method_resize.shape[2]//f ] )
reduced_opt_array_factor_8_method_resize = resize(tight_opt_array_factor_8_method_resize, revised_size, order = 0)

f = 4
revised_size = tuple( [tight_opt_array_factor_4_method_resize.shape[0]//f, tight_opt_array_factor_4_method_resize.shape[1]//f, tight_opt_array_factor_4_method_resize.shape[2]//f ] )
reduced_opt_array_factor_4_method_resize = resize(tight_opt_array_factor_4_method_resize, revised_size, order = 0)


np.save('reduced_opt_array_factor_8_method_zoom.npy',reduced_opt_array_factor_8_method_zoom)
np.save('reduced_opt_array_factor_4_method_zoom.npy',reduced_opt_array_factor_4_method_zoom)
np.save('reduced_opt_array_factor_8_method_resize.npy',reduced_opt_array_factor_8_method_resize)
np.save('reduced_opt_array_factor_4_method_resize.npy',reduced_opt_array_factor_4_method_resize)


