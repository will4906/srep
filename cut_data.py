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
    cut data to odd number trial with even numbers rotate
    cut data to even number trial with odd number rotate
    :return:
    '''
    data_path = os.path.join('.cache', 'dba', 'data')
    mat_list = os.listdir(data_path)
    for mat_name in mat_list:
        trial_index = int(mat_name.split('.')[0].split('-')[-1])
        mat_path = os.path.join(data_path, mat_name)
        mat = scipy.io.loadmat(mat_path)
        new_mat = None
        for index, frame in enumerate(mat.get('data')):
            if trial_index % 2 != 0:
                first = np.append(frame, frame[: 32])
                three = np.append(frame[-16:], np.append(frame, frame[: 16]))
                five = np.append(frame[-32:], frame)
                seven = np.append(frame[-48:], frame[: -16])
                nine = np.append(frame[-64:], frame[: -32])
                eleven = np.append(frame[-80:], frame[: -48])
                thirteen = np.append(frame[-96:], frame[: -64])
                fifteen = np.append(frame[-112:], frame[: -80])
                whole = np.vstack((
                    first, np.vstack((
                        three, np.vstack((
                            five, np.vstack((
                                seven, np.vstack((
                                    nine, np.vstack((
                                        eleven, np.vstack((
                                            thirteen, fifteen
                                        ))
                                    ))
                                ))
                            ))
                        ))
                    ))
                ))
            else:
                two = np.append(frame[-8:], np.append(frame, frame[: 24]))
                four = np.append(frame[-24:], np.append(frame, frame[: 8]))
                six = np.append(frame[-40:], frame[: -8])
                eight = np.append(frame[-56:], frame[: -24])
                ten = np.append(frame[-72:], frame[: -40])
                twelve = np.append(frame[-88:], frame[: -56])
                fortheen = np.append(frame[-104:], frame[: -72])
                sixteen = np.append(frame[-120:], frame[: -88])
                whole = np.vstack((
                    two, np.vstack((
                        four, np.vstack((
                            six, np.vstack((
                                eight, np.vstack((
                                    ten, np.vstack((
                                        twelve, np.vstack((
                                            fortheen, sixteen
                                        ))
                                    ))
                                ))
                            ))
                        ))
                    ))
                ))
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
