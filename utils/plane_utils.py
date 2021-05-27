from matplotlib.pyplot import flag
import numpy as np
import random

from numpy.lib.function_base import angle

def get_point2plane_distance(points, plane):
    A, B, C, D = plane
    return np.abs((A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)) / np.sqrt(A * A + B * B + C * C)

def get_plane_by_lstsq(points):
    """return Ax + By + Cz + D = 0
    """
    if len(points) == 3: #三个点直接计算平面方程
        vector1 = points[0] - points[1]
        vector2 = points[0] - points[2]
        normal = np.cross(vector1, vector2)
        A, B, C = tuple(normal)
        D = (-1) * (normal[0] * points[0][0] + normal[1] * points[0][1] + normal[2] * points[0][2])
        return A, B, C, D
    else: #多个点使用最小二乘法拟合 x_vec = (A_mtx^T * A_mtx)^(-1) * A_mtx^T * b_vec
        #创建系数矩阵A 
        X = np.expand_dims(points[:, 0], axis=1)
        Y = np.expand_dims(points[:, 1], axis=1)
        Z = np.expand_dims(points[:, 2], axis=1)
        ones = np.ones((len(points), 1))

        A_mtx = np.matrix(np.concatenate([Z, Y, ones], axis=1))
        b_vec = X

        x_vec = (A_mtx.T * A_mtx).I * A_mtx.T * b_vec

        A = 1
        B = float(x_vec[1]) * (-1)
        C = float(x_vec[0]) * (-1)
        D = float(x_vec[2]) * (-1)

        return A, B, C, D

def get_plane_by_ransac(points, max_iter, sigma, valid_ratio):
    iter = 0
    best_plane = None
    best_error_num = len(points) + 1

    while(iter < max_iter):
        sample_points_index = random.sample(range(len(points)), 6)
        sample_points = points[sample_points_index]
        test_plane = get_plane_by_lstsq(sample_points)

        test_sigma = get_point2plane_distance(points, test_plane)
        test_error_num = len(test_sigma[test_sigma > sigma])
        # if len(test_sigma[test_sigma > sigma * 3]):
        #     print("too large", len(test_sigma[test_sigma > sigma * 3]) / len(points))

        if test_error_num < best_error_num:
            best_error_num = test_error_num
            best_plane = test_plane

        if best_error_num < len(points) * (1 - valid_ratio):
            break

        iter += 1

    # print(iter, best_error_num)

    return best_plane, iter

def get_plane2Zaxis_angle(plane):
    A, B, C, D = plane
    return np.abs(np.arcsin(C / np.sqrt(A * A + B * B + C * C)))
