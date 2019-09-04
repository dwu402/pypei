import casadi as ca
import numpy as np

def basis_functions(x, k=3):
    """Returns the B spline basis functions over a set of knots, x, of order k"""

    knots = np.concatenate(([x[0]]*k, x, [x[-1]]*k))
    n = len(knots) - k - 1
    basis = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1
        basis.append(ca.Function.bspline('basis' + str(i), [knots], c, [k], 1, dict()))

    return basis

def map_on(mapper, iterable):
    """Returns a numpy array version of a map"""
    return np.array(list(map(mapper, iterable)))

def basis_matrix(x, basis):
    """Returns the basis matrix at collocation points x for some B spline basis"""

    return np.array([[b(t) for b in basis] for t in x])

def diff(basis_function):
    return lambda t: basis_function.jacobian()(t, basis_function(t))

def choose_knots(x, num_knots):
    """ Default knot chooser """
    indexing = np.linspace(0, len(x)-1, num_knots)
    return [x[int(idx)] for idx in indexing]
