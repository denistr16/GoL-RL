import numpy as np


def glider():
    return np.array([[0, 0, 1],
                     [1, 0, 1],
                     [0, 1, 1]])


def gosper_glider_gun():
    pattern_data = np.zeros(11 * 38).reshape(11, 38)
    pattern_data[5][1] = pattern_data[5][2] = 1
    pattern_data[6][1] = pattern_data[6][2] = 1

    pattern_data[3][13] = pattern_data[3][14] = 1
    pattern_data[4][12] = pattern_data[4][16] = 1
    pattern_data[5][11] = pattern_data[5][17] = 1
    pattern_data[6][11] = pattern_data[6][15] = pattern_data[6][17] = pattern_data[6][18] = 1
    pattern_data[7][11] = pattern_data[7][17] = 1
    pattern_data[8][12] = pattern_data[8][16] = 1
    pattern_data[9][13] = pattern_data[9][14] = 1

    pattern_data[1][25] = 1
    pattern_data[2][23] = pattern_data[2][25] = 1
    pattern_data[3][21] = pattern_data[3][22] = 1
    pattern_data[4][21] = pattern_data[4][22] = 1
    pattern_data[5][21] = pattern_data[5][22] = 1
    pattern_data[6][23] = pattern_data[6][25] = 1
    pattern_data[7][25] = 1

    pattern_data[3][35] = pattern_data[3][36] = 1
    pattern_data[4][35] = pattern_data[4][36] = 1

    return pattern_data


gliders = {"glider": glider(),
            "gosper_glider_gun": gosper_glider_gun()}
