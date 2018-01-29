import numpy as np
import os
import scipy.io 

data_path = os.path.join('.cache', 'dba', 'data')
mat_list = os.listdir(data_path)
for mat_name in mat_list:
    mat_path = os.path.join(data_path, mat_name)
    mat = scipy.io.loadmat(mat_path)

    for index, frame in enumerate(mat.get('data')):
        first = np.append(frame[-16:], frame)
        whole = np.append(first, frame[0:16])
        if index == 0:
            new_mat = whole
        else:
            new_mat = np.vstack((new_mat, whole))

    mat.__setitem__('data', new_mat)
    save_path = os.path.join('.cache2', 'dba', 'data', mat_name)
    scipy.io.savemat(save_path, mat)
