""" Interface for creating CasADi symbolics for weighted least squares """
import numpy as np
import casadi as ca
from typing import Iterable

def replace_nan(data_obj):
    """ Replaces nans with zeros, additionally returns locations of nans 
    
    Returns
    -------
    [0]: data_obj with nans as zeros
    [1]: list of indices that nans were at [(i,j), (i,j), ...]
    """

    nan_inds = list(zip(*np.indices(data_obj.shape)[:, np.isnan(data_obj)]))
    return np.nan_to_num(data_obj), nan_inds

class Objective():
    """ Contains the objective function """
    def __init__(self, config=None):
        self.ys = []
        self.y0s = []
        self._y0s = []
        self.Ls = []
        self._Ls = []
        self._ws = []

        self.objective_function = None
        self.log_likelihood = None

        if config:
            self.make(config)

    def __str__(self):
        y_szs = [x.shape for x in self.ys]
        return f"pypei Objective {y_szs}"

    @staticmethod
    def _DATAFIT(model, obsv_fn=lambda x, p: x):
        """ Default data fit objective value

        Assuming form ||y_0 - g(x, p)||^2, returns g(x, p)
        for model state x and model parameters p
        """
        return obsv_fn(model.xs, model.ps).reshape((-1, 1))

    @staticmethod
    def _MODELFIT(model, dt=True):
        """ Default model fit objective value

        Assuming form ||Dx- f(x,p)||^2, returns Dx-f(x,p)
        """
        if dt:
            return ((model.xdash - model.model(model.observation_times,
                                                *model.cs, *model.ps)
                    )*np.sqrt(np.gradient(model.observation_times))
                    ).reshape((-1, 1))
        else:
            return (model.xdash - model.model(model.observation_times,
                                            *model.cs, *model.ps)
                   ).reshape((-1, 1))

    @staticmethod
    def _autoconfig_data(data: np.array, select: list = None):
        """ Generates objective config components for data

        Creates an observation function that removes effects of nans in data"""
        config = {'sz': data.shape}
        if not select:
            config['obs_fn'] = lambda x: (1-np.isnan(data).astype(float)) * x
        else:
            config['obs_fn'] = lambda x: ((1-np.isnan(data).astype(float)) * x)[:, select]
        return config

    @staticmethod
    def _autoconfig_L(data, auto=False, sigma=None):
        """ Generates default configuration for L matrix

        Default L matrix is of form 1/sigma * I

        Options
        -------
        auto (bool): whether ot not configure the variance to be estimated
        Assumes form L = 1/sigma * I (where sigma is to be estimated)
        sigma (casadi SX): symbolic representing the variance to be estimated
        """
        if auto:
            config = Objective._autoconfig_autoL(data, sigma)
        else: # default path
            config = Objective._autoconfig_constL(data)
        return config

    @staticmethod
    def _autoconfig_autoL(data, sigma):
        """ L configuration for automated optimisation of variance of form

        L = 1/sigma * I
        """
        return {
                'depx': True,
                'x': 1 / sigma * ca.SX.eye(np.prod(data.shape)),
                'iden': True,
                'balance': False,
               }

    @staticmethod
    def _autoconfig_constL(data):
        """ L configuration for optimisation under given L, where

        L = 1/sigma * I
        """
        return {
                'n': np.prod(data.shape),
                'iid': True,
                'balance': False
               }

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
        iid: (bool) whether or not $L_F$ takes the form $L_F = s*I$
        numL: (int) number of free parameters in $L_F$
        struct: (iterable) locations of the free parameters
        n: (int) size of $L_F$
        balance: (bool) whether or not to divide obj fn component by size of L

        Further Options
        ---------------

        struct (L) : <list of> index by i
        1. <list>
            represents a list of indices to insert the i-th symbol
            <tuple>
                Used as an index into an n by n matrix to place the i-th symbol.
            <int>
                Location on the diagonal of an n by n matrix to place i-th symbol
        2. <dict> 
            represents a diagonal structure to place copies of the i-th symbol
            i0 : upper left corner index
            n : number of rows
        """
        assert len(config['L']) == len(config['Y'])
        # create L matrix symbolics
        for i, L in enumerate(config['L']):
            if 'depx' in L and L['depx'] and 'x' in L:
                L_base = L['x']
            else:
                L_base = ca.SX.eye(L['n'])
            if 'balance' in L and L['balance']:
                self._ws.append(1.0/L_base.shape[0])
            else:
                self._ws.append(1)
            if "iden" in L and L['iden']:
                self._Ls.append(L_base)
                continue
            if 'iid' in L and L['iid']:
                Lobj = ca.SX.sym(f'L_{i}')
                self.Ls.append(Lobj)
                self._Ls.append(L_base@Lobj)
            elif 'numL' in L and 'struct' in L and L['numL'] and L['struct']:
                Lobj = ca.SX.sym(f'L_{i}', L['numL'])
                self.Ls.append(Lobj)
                _L = ca.SX(L['n'], L['n']) # structural zero matrix
                for Li, info in zip(Lobj.nz, L['struct']):
                    if 'n' in info: # dict-int style
                        i0, n = (info['i0'], info['n'])
                        _L[i0:i0+n, i0:i0+n] = Li * ca.SX.eye(n)
                    elif 'ns' in info: # dict-list style
                        i0s, ns = (info['i0s'], info['ns'])
                        for i0, n in zip(i0s, ns):
                            _L[i0:i0+n, i0:i0+n] = Li * ca.SX.eye(n)
                    else: # index list style
                        for i in info:
                            if isinstance(i, int):
                                _L[i,i] = Li
                            else:
                                i, j = i
                                _L[i, j] = Li
                self._Ls.append(L_base@_L)
            else:
                # default is a completely free L matrix
                Lobj = ca.SX.sym(f'L_{i}', L['n'], L['n'])
                self.Ls.append(Lobj)
                self._Ls.append(L_base@Lobj)
        # create Y0 (data) symbolics and Y symbolics
        for i, Y in enumerate(config['Y']):
            if 'unitary' in Y and Y['unitary']:
                y0i = ca.SX.sym(f'Y0_{i}')
                self.y0s.append(y0i)
                self._y0s.append((y0i * ca.SX.ones(Y['sz'])).reshape((-1, 1)))
            else:
                y0i = ca.SX.sym(f'Y0_{i}', *Y['sz'])
                self.y0s.append(y0i)
                self._y0s.append(y0i.reshape((-1, 1)))
            self.ys.append(Y['obs_fn'])
        # assemble objective function
        self.assemble_objective()

    def assemble_objective(self):
        """ (Re)builds the objective function from L, data and model components """
        self.objective_function = sum(w * ca.sumsqr(L@(y0-y))
                                      for L, y0, y, w in zip(self._Ls, self._y0s, self.ys, self._ws))
        self.log_likelihood = sum(w * ca.sumsqr(L@(y0-y))
                                  - 2*ca.sum1(ca.log(ca.diag(L)))
                                  for L, y0, y, w in zip(self._Ls, self._y0s, self.ys, self._ws))

    def obj_fn(self, i):
        """ Returns the nth objective function object """
        return self._ws[i] * ca.sumsqr(self._Ls[i]@(self._y0s[i]-self.ys[i]))

    def us_obj_fn(self, i):
        """ Returns the nth objective function object, without covariance scaling """
        return ca.sumsqr(self._y0s[i]-self.ys[i])

    def obj_comp(self, i):
        """ Returns the components of the nth objective function object"""
        return self._Ls[i]@(self._y0s[i]-self.ys[i])

    def us_obj_comp(self, i):
        """ Returns the components of the nth objective function object, without covariance scaling """
        return self._y0s[i] - self.ys[i]

def ignore_nan(data, casobj=None):
    """ Removes nans and flattens data and corresponding CasADi object

    Returns
    -------
    indexer: array of bool
        Truthy table of where data is finite (non nan or inf)
    <tuple>
        Flattened and nan-stripped data (and casobj if provided)
    """
    indexer = np.isfinite(data)

    if casobj is None:
        return indexer, (data[indexer],)
    else:
        assert data.shape == casobj.shape, f"{data.shape} not the same as {casobj.shape}"
        # CasADi objects do not support logical indexing
        # CasADi objects flatten column-first (with .nz), cf np.array which flatten row-first
        igcasobj = ca.vcat([oi for oi,z in zip(casobj.T.nz, indexer.flatten()) if z])
        return indexer, (data[indexer], igcasobj)

def L_via_data(config, data):
    indexer, _ = ignore_nan(data)

    accum = 0
    config['numL'] = 0
    config['struct'] = []
    for data_col in indexer:
        n = sum(data_col)
        config['struct'].append({'n': n, 'i0': accum})
        accum += n
        config['numL'] += 1

    config['iid'] = False

def map_order_to_L_struct(order, n_sz, inherent_order=None):
    """
    Maps a given structured order of quantities with length n_sz to
    an appropriate struct field of an L config.

    If inherent order is not given, assumes order contains numerical
    indices (0, 1, ...)

    Example
    -------
    map_order_to_L_struct(['AB', 'CF', 'E', 'D'], 50, 'ABCDEF')
    >> [
        {'ns': [50, 50], 'i0s': [0, 50]},
        {'ns': [50, 50], 'i0s': [100, 250]},
        {'n': 50, 'i0': 200},
        {'n': 50, 'i0': 150}, 
    ]
    """

    if inherent_order is None:
        find = lambda i: i
    else:
        find = lambda i: inherent_order.index(i)

    struct = list()
    for x in order:
        elem = dict()
        if isinstance(x, Iterable) and len(x) > 1:
            elem['ns'] = []
            elem['i0s'] = []
            for i in x:
                elem['ns'].append(n_sz)
                elem['i0s'].append(n_sz*find(i))
        else:
            elem['n'] = n_sz
            elem['i0'] = n_sz*find(x)
        struct.append(elem)
    return struct

