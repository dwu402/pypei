"""A utility to extract BSpline functions and their time derivatives from the Casadi interface of Python"""
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

def cross_map(iter_mapper, iterable):
    """Maps each map in iter_mapper on each iterable"""
    return [list(map(im, iterable)) for im in iter_mapper]

def basis_matrix(x, basis):
    """Returns the basis matrix at collocation points x for some B spline basis"""
    return np.array(cross_map(basis, x)).T

def diff(basis_function):
    """Creates a function that can evaluate the derivative (in time) of a basis function"""
    return lambda t: basis_function.jacobian()(t, basis_function(t))

def diff_list(basis):
    """Returns the list of time derivatives for all basis functions in basis"""
    return list(map(diff, basis))

def diff_matrix(x, basis):
    """Returns the basis time derivative matrix at collocation points x for some B spline basis"""
    return np.array(cross_map(diff_list(basis), x)).T

def choose_knots(x, num_knots):
    """ Uniform knot chooser """
    indexing = np.linspace(0, len(x)-1, num_knots)
    return [x[int(idx)] for idx in indexing]
