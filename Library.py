import numpy as np


def copy_2d_array(array_2d):
    new_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for x1 in range(3):
        for y1 in range(3):
            new_list[x1][y1] = array_2d[y1][x1]
    return new_list


def conv_array(array_2d):
    new_list = []
    for x1 in range(7):
        for y1 in range(7):
            new_list.append(array_2d[y1][x1] * 0.5)
    return new_list


def to_np_array(list):
    return np.array(list).flatten()
