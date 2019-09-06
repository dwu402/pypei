import numpy as np
import casadi as ca
from functions import casbasis


class Model():
    def __init__(self, configuration=None):
        self.ts = None
        self.cs = None
        self.xs = None
        self.ps = None
        self.n = 0
        self.K = 0
        self.s = 0
        self.observation_times = None
        self.basis = None
        self.basis_jacobian = None
        self.getx = None
        self.xdash = None
        self.model = None

        if configuration is not None:
            self.generate_model(configuration)

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
        if configuration['knot_function'] is None:
            knots = casbasis.choose_knots(self.observation_times, self.K-2)
        else:
            knots = configuration['knot_function'](self.observation_times, self.K-2, configuration['dataset'])
        basis_fns = casbasis.basis_functions(knots)
        self.basis = ca.vcat([b(self.ts) for b in basis_fns]).reshape((self.n, self.K))

        # create the objects that define the smooth, model parameters
        self.cs = [ca.MX.sym("c_"+str(i), self.K, 1) for i in range(self.s)]
        self.xs = [self.basis@ci for ci in self.cs]
        self.ps = [ca.MX.sym("p_"+str(i)) for i in range(n_ps)]

        # model function derived from input model function
        self.model = ca.Function("model",
                                 [self.ts, *self.cs, *self.ps],
                                 [ca.hcat(configuration['model'](self.ts, self.xs, self.ps))])

    def get_x(self, *cs):
        """ Exposes calculation of trajectory """
        if self.getx is None:
            self.getx = ca.Function("getx",
                                    [self.ts, *self.cs],
                                    self.xs)
        return self.getx(self.observation_times, *cs)


    # We calculate expensive stuff later when called
    def get_basis_jacobian(self):
        """ Lazy evaluation of the jacobian of the basis """
        if self.basis_jacobian is None:
            self.basis_jacobian = ca.vcat(
                [ca.diag(ca.jacobian(self.basis[:, i], self.ts)) for i in range(self.K)]
            ).reshape((self.n, self.K))
        return self.basis_jacobian

    def get_xdash(self):
        """ Lazy evaluation of the smooth derivatives """
        if self.xdash is None:
            self.xdash = self.get_basis_jacobian()@ca.hcat(self.cs)
        return self.xdash
