""" Interface for creating CasADi symbolics for weighted least squares """
import numpy as np
import casadi as ca

class Objective():
    """ Contains the objective function """
    def __init__(self, config=None):
        self.ys = []
        self.y0s = []
        self._y0s = []
        self.Ls = []
        self._Ls = []

        self.objective_function = None

        if config:
            self.make(config)

    @staticmethod
    def _DATAFIT(model, obsv_fn=lambda x, p: x):
        """ Default data fit objective value

        Assuming form ||y_0 - g(x, p)||^2, returns g(x, p)
        for model state x and model parameters p
        """
        return obsv_fn(model.xs, model.ps).reshape((-1, 1))

    @staticmethod
    def _MODELFIT(model):
        """ Default model fit objective value

        Assuming form ||Dx- f(x,p)||^2, returns Dx-f(x,p)
        """
        return (model.xdash - model.model(model.observation_times, 
                                          *model.cs, *model.ps)
               ).reshape((-1, 1))

    @staticmethod
    def _autoconfig_data(data: np.array, select: list = None) -> (dict, np.array):
        """ Generates objective config components for data

        Creates an observation function that removes effects of nans in data"""
        config = {'sz': data.shape}
        if not select:
            config['obs_fn'] = lambda x: (1-np.isnan(data).astype(float)) * x
        else:
            config['obs_fn'] = lambda x: ((1-np.isnan(data).astype(float)) * x)[:, select]
        return config

    @staticmethod
    def _autoconfig_L(data):
        """ Generates default configuration for L matrix

        Default L matrix is of form 1/sigma * I
        """
        config = {'n': np.prod(data.shape), 'diag': True, 'balance': False}
        return config

    def make(self, config):
        """ Make the objective function given the configuration options

        The configuration consists of two keys: Y and L, which configure
        the data based objects and L matrices respectively

        The objective function is of the form
        $$
        H(x|L, y_0) = \\sum_{i=0}^{N} || L_i({y_0}_i - f_i(x)) ||^2
        $$

        $L_i$ is further assumed to have the form:
        $$
        L_i = L_FL_D(x)
        $$
        where $L_F$ is a fixed portion and $L_D$ is an x-dependent portion.

        $L_i$ are also modelled as the Cholesky factors of the inverse covariance matrices,
        so $L_i$ is expected to be triangular.

        Configuration Options
        ---------------------
        Y

        sz: (tuple of floats) Size of the data (m,n)
        obs_fn:  casadi object representing f_i(x)

        L

        depx: (bool) whether or not L has an x-dependent portion
        x: the form on $L_D(x)$. Required if depx is True
        iden: (bool) whether or not $L_F$ is identity, overrides diag option below
        diag: (bool) whether or not $L_F$ takes the form $L_F = s*I$
        n: (int) size of $L_F$
        balance: (bool) whether or not to divide obj fn component by size of L
        """
        assert len(config['L']) == len(config['Y'])
        # create L matrix symbolics
        for i, L in enumerate(config['L']):
            if 'depx' in L and L['depx'] and 'x' in L:
                L_base = L['x']
            else:
                L_base = ca.SX.eye(L['n'])
            if 'balance' in L and L['balance']:
                L_base /= ca.sqrt(L_base.shape[0])
            if "iden" in L and L['iden']:
                self._Ls.append(L_base)
                continue
            if 'diag' in L and L['diag']:
                Lobj = ca.SX.sym(f'L_{i}')
                self.Ls.append(Lobj)
                self._Ls.append(L_base@Lobj)
            else:
                Lobj = ca.SX.sym(f'L_{i}', L['n'], L['n'])
                self.Ls.append(L_base@Lobj)
                self._Ls.append(L_base@Lobj)
        # create Y0 (data) symbolics and Y symbolics
        for i, Y in enumerate(config['Y']):
            if 'unitary' in Y and Y['unitary']:
                y0i = ca.SX.sym(f'Y0_{i}')
                self.y0s.append(y0i)
                self._y0s.append((y0i * ca.SX.ones(Y['sz'])).T.reshape((-1, 1)))
            else:
                y0i = ca.SX.sym(f'Y0_{i}', *Y['sz'])
                self.y0s.append(y0i)
                self._y0s.append(y0i.T.reshape((-1, 1)))
            self.ys.append(Y['obs_fn'])
        # assemble objective function
        self.assemble_objective()

    def assemble_objective(self):
        self.objective_function = sum(ca.sumsqr(L@(y0-y)) 
                                      - 2*ca.sum1(ca.log(ca.diag(L)))
                                      for L, y0, y in zip(self._Ls, self._y0s, self.ys))

    def obj_fn(self, i):
        """ Returns the nth objective function object """
        return ca.sumsqr(self._Ls[i]@(self.y0s[i]-self.ys[i]))

    def us_obj_fn(self, i):
        return ca.sumsqr(self.y0s[i]-self.ys[i])