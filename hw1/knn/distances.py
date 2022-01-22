import numpy as np

def euclidean_distance(x: np.array, y: np.array) -> np.array:
    # D = (y - x)^2 = y^2 + x^2 - 2*y*x
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    m, d = x.shape
    n, d_y = y.shape

    if d != d_y:
        raise TypeError(f'Inconsistent dimensions: D_x = {d}, D_y = {d_y}')

    x_squares = (x * x).dot(np.ones((d, n)))
    y_squares = (y * y).dot(np.ones((d, m))).T
    xy = x.dot(y.T)

    squared_distances_matrix = x_squares + y_squares - 2 * xy
    return np.sqrt(squared_distances_matrix)
    # return squared_distances_matrix

def cosine_distance(x: np.array, y: np.array) -> np.array:
    # x * y = |x|*|y|*cos(x, y) => cos(x, y) = (x * y) / (|x| * |y|)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    m, d = x.shape
    n, d_y = y.shape

    if d != d_y:
        raise TypeError(f'Inconsistent dimensions: D_x = {d}, D_y = {d_y}')

    xy = x.dot(y.T)
    x_squared = (x * x).dot(np.ones((d, n)))
    y_squared = (y * y).dot(np.ones((d, m))).T
    return xy / np.sqrt(x_squared * y_squared)
