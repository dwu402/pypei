from inspect import signature
from functools import wraps
from scipy.linalg import block_diag as bd
import numpy as np
import casadi as ca

def knot_fn(ts, n, dataset):
    """ A curvature based knot location selection function
    Inputs:
    ts - fine time grid
    n - number of knots
    dataset - ['y'] contains the data
              ['t'] contains the time gridding of data
    """
    y = dataset['y'].flatten()
    times = dataset['t'].flatten()
    diffs = np.gradient(np.gradient(y, times), times)
    ntimes = len(times)
    importance = sorted(range(ntimes), key=lambda i: np.abs(diffs[i]), reverse=True)
    if n <= ntimes:
        # ensure that 0 and -1 are in the knot vector
        temp_knots = importance[:n]
        if 0 in temp_knots:
            temp_knots.remove(0)
        if ntimes-1 in temp_knots:
            temp_knots.remove(ntimes-1)
        knot_indices = [0] + sorted(temp_knots[:n-2]) + [-1]

        # match the times for knots
        corresponding_times = times[knot_indices]
        # align along fine grid (optional)
        return [min(ts, key=lambda t: np.abs(t-tk)) for tk in corresponding_times]
    else:
        # determine which time points to refine
        copies = (n//ntimes)*np.ones(ntimes)
        copies[importance[:(n%ntimes)]] += 1
        copies = [int(j) for j in copies]
        # compute the number of knot points in each gap
        kgn = [int(copies[0]-1)]
        for ci in copies[1:-1]:
            m = int(ci//2)
            kgn[-1] += m
            kgn.append(m)
        kgn[-1] += copies[-1]-1
        # select knots to keep, always keep end knots
        keep = [int(ci%2) for ci in copies]
        keep[-1] = 1
        knots = [times[0]]
        # construct knot locations
        for gapn, k0, k1, x0, x1, c in zip(kgn, keep[:-1], keep[1:], times[:-1], times[1:], copies[1:]):
            step = 2*(x1-x0)/(2*gapn+k0+k1)
            cands = np.arange(x0-(1-k0)*step/2, x1+(3-k1)*step/2, step)
            frag = cands[1:(gapn+1+k1)]
            knots.extend(frag)
        return sorted(knots)

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
    """ Flattens all elements of a list """
    return [
        arg.reshape((-1, 1)) for arg in args
    ]

def _filter_arguments(function, arguments, remove=[]):
    """ Filters kwargs for a given function 
    from https://stackoverflow.com/questions/55590419/filter-keyword-arguments-when-calling-function
    """
    return {k:a for k,a in arguments.items() 
            if k in signature(function).parameters and k not in remove}

def func_kw_filter(func):
    @wraps(func)
    def filtered_func(*args, **kwargs): 
        return func(*args, **_filter_arguments(func, kwargs)) 
    return filtered_func 
