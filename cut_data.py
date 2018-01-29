import logging
import numpy as np
import os
import scipy.io

logging.getLogger().setLevel(logging.DEBUG)


def add_two_rows():
    '''
    change all files to 15, 16, 1, 2, ..., 15, 16, 1, 2
    :return:
    '''
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


def add_a_period():
    '''
    1, 2, ..., 15, 16, 1, 2, 3, 4
    16, 1, 2, ..., 15, 16, 1, 2, 3
    15, 16, 1, 2, ..., 15, 16, 1, 2
    14, 15, 16, 1, 2, ..., 15, 16, 1
    13, ..., 16, 1, 2, ..., 15, 16
    12, ..., 16, 1, 2, ..., 15

    15, 16, 1, 2, ..., 15, 16, 1, 2
    14, 15, 16, 1, 2, ..., 15, 16, 1
    13, 14, 15, 16, 1, 2, ..., 15, 16
    12, 13, 14, 15, 16, 1, 2, ..., 15
    11, ..., 16, 1, 2, ..., 14
    10, ..., 16, 1, 2, ..., 13
    9, ..., 16, 1, 2, ..., 12
    8, ..., 16, 1, 2, ..., 11
    7, ..., 16, 1, 2, ..., 10
    6, ..., 16, 1, 2, ..., 9
    5, ..., 16, 1, 2, ..., 8
    4, ..., 16, 1, 2, ..., 7
    3, ..., 16, 1, 2, ..., 6
    2, ..., 16, 1, 2, ..., 5
    1, ..., 16, 1, 2, ..., 4
    16, 1, 2, ..., 16, 1, 2, 3
    :return:
    '''
    data_path = os.path.join('.cache', 'dba', 'data')
    mat_list = os.listdir(data_path)
    for mat_name in mat_list:
        mat_path = os.path.join(data_path, mat_name)
        mat = scipy.io.loadmat(mat_path)

        for index, frame in enumerate(mat.get('data')):
            first = np.append(frame, frame[: 32])
            second = np.append(frame[-8:], np.append(frame, frame[: 24]))
            third = np.append(frame[-16:], np.append(frame, frame[: 16]))
            four = np.append(frame[-24:], np.append(frame, frame[: 8]))
            five = np.append(frame[-32:], frame)
            six = np.append(frame[-40:], frame[: -8])
            seven = np.append(frame[-48:], frame[: -16])
            eight = np.append(frame[-56:], frame[: -24])
            night = np.append(frame[-64:], frame[: -32])
            ten = np.append(frame[-72:], frame[: -40])
            eleven = np.append(frame[-80:], frame[: -48])
            twelve = np.append(frame[-88:], frame[: -56])
            thirteen = np.append(frame[-96:], frame[: -64])
            fortheen = np.append(frame[-104:], frame[: -72])
            fifteen = np.append(frame[-112:], frame[: -80])
            sixteen = np.append(frame[-120:], frame[: -88])

            whole = np.vstack((first,
                               np.vstack((
                                   second, np.vstack((
                                       third, np.vstack((
                                           four, np.vstack((
                                               five, np.vstack((
                                                   six, np.vstack((
                                                       seven, np.vstack((
                                                           eight, np.vstack((
                                                               night, np.vstack((
                                                                   ten, np.vstack((
                                                                       eleven, np.vstack((
                                                                           twelve, np.vstack((
                                                                               thirteen, np.vstack((
                                                                                   fortheen, np.vstack((
                                                                                       fifteen, sixteen
                                                                                   ))
                                                                               ))
                                                                           ))
                                                                       ))
                                                                   ))))
                                                           ))
                                                       ))
                                                   ))
                                               ))
                                           ))))))))))
            if index == 0:
                new_mat = whole
            else:
                new_mat = np.vstack((new_mat, whole))
        mat.__setitem__('data', new_mat)
        save_path = os.path.join('.cache2', 'dba', 'data', mat_name)
        scipy.io.savemat(save_path, mat)
        logging.info(save_path)

if __name__ == '__main__':
    add_a_period()
