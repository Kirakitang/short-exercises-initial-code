'''
Short Exercises #5
'''

import numpy as np


def compute_matching(x, y):
    """
    Returns a new array which is "true" everywhere x == y and
    false otherwise.

    Note this is simply the Numpy version of the same function
    in SE2 but should be substantially simpler.

    Input:
        x: n-dimensional array
        y: n-dimensional array

    Returns: Boolean-valued n-dimensional array with the same shape as
             x and y
    """

    # YOUR CODE HERE
    a = np.array(x)
    b = np.array(y)
    # 对于二维的数组，数组形状相同时才可以判断(除非可播放)
    # 如一个(2,3)的数组和一个(2,2)的数组判断时会报错
    c = (a == b) # Numpy Broadcast
    # Replace None with an appropriate return value
    return c


def compute_matching_indices(x, y):
    """
    Returns a new array consisting of the indices where
    x == y.

    Note this is simply the Numpy version of the same function
    in SE2 but should be substantially simpler.

    Input:
        x: 1-dimensional array
        y: 1-dimensional array

    Returns: a sorted array of the indices where x[i] == y[i]

    Note that the returned array must be one-dimensional!

    """

    # YOUR CODE HERE
    a = np.array(x)
    b = np.array(y)
    bool_equal = (a == b)
    grouped_index = np.argwhere(bool_equal)
    row = grouped_index.shape[0]
    col = grouped_index.shape[1]
    ans_index = grouped_index.reshape(row*col)
    # Replace None with an appropriate return value
    return ans_index
# x = [1,2,3]
# y = [1,3,3]
# print(compute_matching_indices(x,y))


def powers(N, p):
    """
    Return the first N powers of p. For example:
    powers(5, 2) --> [1, 2, 4, 8, 16]
    powers(5, 4) --> [1, 4, 16, 64, 256]

    Input:
       N: number of powers to return
       p: base that we are raising to the given powers

    Returns: an array consisting of powers of p
    """

    # YOUR CODE HERE
    ans = np.logspace(0, (N-1), N, base = p)
    # Replace None with an appropriate return value
    return ans
# N = 2
# p = 1
# print(powers(N,p))

def clip_values(x, min_val=None, max_val=None):
    """
    Return a new array with the values clipped.

    If min_val is set, all values < min_val will be set to min_val
    If max_val is set, all values > max_val will be set to max_val

    Remember to return a new array, NOT to modify the input array.

    Inputs:
        x: the n-dimensional array to be clipped
        min_val : the minimum value in the returned array (if not None)
        max_val : the maximum value in the returned array (if not None)

    returns: an array with the same dimensions of X with values clipped
             to (min_val, max-val)

    """

    # YOUR CODE HERE
    if min_val != None:
        x = np.where(x>=min_val, x, min_val)
    if max_val != None:
        x = np.where(x<=max_val, x, max_val)
    ans = x.flatten()
    # Replace None with an appropriate return value
    return ans


def find_closest_value(x):
    """
    Returns the index and corresponding value in the one-dimensional
    array x that is closest to the mean

    Examples:
    find_closest_value(np.array([1.0, 2.0, 3.0])) -> (1, 2.0)
    find_closest_value(np.array([5.0, 1.0, 8.0])) -> (0, 5.0)

    Inputs:
        x: 1-dimensional array of values

    Returns: the index and the scalar value in x that is
        closest to the mean

    """

    # YOUR CODE HERE
    mean = np.mean(x)
    dif = x - mean
    abs_dif = np.maximum(dif, -dif)
    index = np.argmin(abs_dif)
    value = x[index]
    # Replace None with an appropriate return value
    return (index, value)
# x = np.array([1.0, 2.0, 3.0])
# print(find_closest_value(x))

def select_row_col(x, row_idx=None, col_idx=None):
    """
    Select a subset of rows or columns in the two-dimensional array x.

    Inputs:
        x: input two-dimensional array
        row_idx: a list of row index we are selecting, None if not specified
        col_idx: a list of column index we are selecting, None if not specified

    Returns: a two-dimensional array where we have selected based on the
        specified row_idx and col_idx
    """

    # YOUR CODE HERE
    if row_idx != None:
        x = x[row_idx]
    if col_idx != None:
        x = x[..., col_idx]
    # Replace None with an appropriate return value
    return x
# x = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8]])
# print(select_row_col(x, [1,2], [0,2]))
