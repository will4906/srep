import numpy as np
import os
import scipy.io


def load_whole_train_data():
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
    data_path = os.path.join(now_path, '.cache', 'dba', 'data')
    mat_list = os.listdir(data_path)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_index = [1, 3, 5, 7, 9]
    test_index = [2, 4, 6, 8, 10]
    for mat_name in mat_list:
        gesture_index = int(mat_name.split('.')[0].split('-')[1])
        trial_index = int(mat_name.split('.')[0].split('-')[-1])
        if gesture_index < 9:
            mat_path = os.path.join(data_path, mat_name)
            mat = scipy.io.loadmat(mat_path)
            if trial_index in train_index:
                for frame in mat.get('data'):
                    train_x.append(frame)
                    train_y.append(mat.get('gesture')[0][0])
            else:
                for frame in mat.get('data'):
                    test_x.append(frame)
                    test_y.append(mat.get('gesture')[0][0])
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)

    # print(train_x.shape)


if __name__ == '__main__':
    load_whole_train_data()