""" Utility functions """

from scipy.linalg import block_diag as bd
import numpy as np
import casadi as ca

def block_diag(block_size, weights, casadi=False):
    """ Creates a diagonal matrix where block_size entries are identical 

    Input
    -----
    block_size: size of a single block
    weights: list of length n, values of block entries
    casadi: whether to use CasAdi SX symbolics instead of numpy

    Example
    -------
    block_diag(3, [1, 2])
    >>> np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2]
        ])
    """
    if not casadi:
        eye = np.eye(block_size)
    else:
        eye = ca.SX.eye(block_size)
    return bd([eye*w for w in weights])

def flat_squash(*args):
    return [
        arg.reshape((-1,1)) for arg in args
    ]

