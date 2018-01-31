import os
import numpy as np
import scipy.io


def load_single_train_data(base_path, subject_id):
    '''
    加载单人数据
    '''
    subject_id = int(subject_id)
    if subject_id > 9:
        subject_path = base_path + os.sep + '0' + str(subject_id)
    else:
        subject_path = base_path + os.sep + '00' + str(subject_id)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    mat_list = os.listdir(subject_path)
    for mat in mat_list:
        mat_split = mat.split('.')
        if mat_split[-1] == 'mat':
            if int(mat_split[0].split('-')[-1]) % 2 == 1:
                mat_file = scipy.io.loadmat(subject_path + os.sep + mat)
                for frame in mat_file.get('data'):
                    train_x.append(frame)
                    train_y.append(
                        mat_file.get('gesture')[0][0])
            else:
                mat_file = scipy.io.loadmat(subject_path + os.sep + mat)
                for frame in mat_file.get('data'):
                    test_x.append(frame)
                    test_y.append(mat_file.get('gesture')[0][0])
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def load_whole_train_data(base_path):
    pass