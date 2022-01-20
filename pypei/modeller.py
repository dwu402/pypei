""" Interface for CasADi B Spline basis creation """
import numpy as np
import casadi as ca
from .functions import casbasis

class Model():
    """ B-spline basis representation of state 
    
    Configuration Options
    ---------------------
    grid_size : int : number of collocation points
    basis_number : int : number of basis functions
    model : callable : undelying model
    model_form : dict : 
        state : int : number of states in the underlying model
        parameters : int : number of parameters in the underlying model
    time_span : tuple <float> : start and end time of fitting window
    knot_function : callable (optional) : knot location generator based on the data. Defaults to uniformly spaced knots
    dataset : array (optional) : dataset to pass to knot_function
    dphi : callable (optional) : User-input function to generate matrix representing the differential operator. Defaults to using exact derivative from casadi.

    Notes
    -----
    model : has signature model(t, y, p) -> dydt. Inputs are (t) time/independent variable (y) state (p) model parameters
    dphi: has signature dphi(t) -> D_t. Input is a list of collocation times in the fitting window
    """
    def __init__(self, configuration=None):
        self.ts = None
        self.cs = None
        self.xs = None
        self.ps = None
        self.n = 0
        self.K = 0
        self.s = 0
        self.observation_times = None
        self.basis_fns = None
        self.basis = None
        self.basis_jacobian = None
        self.getx = None
        self.xdash = None
        self.x_obsv = None
        self.xdash_obsv = None
        self.model = None

        if configuration is not None:
            self.generate_model(configuration)

    def __str__(self):
        nps = getattr(self.ps, "__len__", lambda : 0)
        return f"pypei Model ({self.s} x {nps()}) -> ({self.K} x {self.n})"

    def generate_model(self, configuration):
        """ Logic to construct a model, and smooth representation on BSpline basis """
        self.n = configuration['grid_size']
        self.K = configuration['basis_number']
        self.s = configuration['model_form']['state']
        n_ps = configuration['model_form']['parameters']

        # setup fine time grid
        self.ts = ca.MX.sym("t", self.n, 1)
        self.observation_times = np.linspace(*configuration['time_span'][:2], self.n)

        # determine knots and build basis functions
        if 'knot_function' not in configuration or configuration['knot_function'] is None:
            knots = casbasis.choose_knots(self.observation_times, self.K-2)
        else:
            knots = configuration['knot_function'](self.observation_times, self.K-2, configuration['dataset'])
        self.basis_fns = casbasis.basis_functions(knots)
        self.basis = ca.vcat([b(self.ts) for b in self.basis_fns]).reshape((self.n, self.K))

        self.tssx = ca.SX.sym("t", self.n, 1)

        # define basis matrix and gradient matrix
        phi = ca.Function('phi', [self.ts], [self.basis])
        self.phi = np.array(phi(self.observation_times))

        if 'dphi' in configuration and configuration['dphi'] is not None:
            self.basis_jacobian = configuration['dphi'](self.observation_times) @ self.phi
        else:
            bjac = ca.hcat([b.jacobian()(self.ts, bt) for b,bt in zip(self.basis_fns, ca.horzsplit(self.basis))])
            self.basis_jacobian = np.array(ca.Function('bjac', [self.ts], [bjac])(self.observation_times))

        # create the objects that define the smooth, model parameters
        self.cs = [ca.SX.sym("c_"+str(i), self.K, 1) for i in range(self.s)]
        self._xs = [self.phi@ci for ci in self.cs]
        self.xs = ca.hcat(self._xs)
        self.xdash = self.basis_jacobian@ca.hcat(self.cs)
        self.ps = [ca.SX.sym("p_"+str(i)) for i in range(n_ps)]

        # model function derived from input model function
        model_fn = configuration.get('model', None)
        if model_fn is not None:
            self.make_model(model_fn)

    def make_model(self, model_fn):
        model_output = [model_fn(self.tssx[i], self.xs[i,:], ca.vcat(self.ps)).reshape((1, -1))
                        for i in range(self.n)]
        self.model = ca.Function("model",
                                 [self.tssx, *self.cs, *self.ps],
                                 [ca.vcat(model_output)])

    def get_x(self, *cs):
        """ Exposes calculation of state from given spline coefficients"""
        if self.getx is None:
            self.getx = ca.Function("getx",
                                    [*self.cs],
                                    self.xs)
        return self.getx(*cs)

    def x_at(self, cs, ts):
        t_phi = np.vstack([b(ts).toarray().flatten() for b in self.basis_fns]).T
        return t_phi@cs

    def all_x_at(self, cs, ts):
        t_phi = np.vstack([b(ts).toarray().flatten() for b in self.basis_fns]).T
        return ca.hcat([t_phi@c for c in cs])