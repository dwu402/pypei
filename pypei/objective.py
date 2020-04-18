import copy
import casadi as ca
import numpy as np
from . import util
from . import modeller

class Objective():
    """ Contains the objective function """
    def __init__(self, config=None):
        self.ys = []
        self.y0s = []
        self.Ls = []
        self._Ls = []

        self.objective_function = None

        if config:
            self.make(config)

    @staticmethod
    def _DATAFIT(model, obsv_fn=lambda x:x):
        return obsv_fn(model.xs).reshape((-1,1))

    @staticmethod
    def _MODELFIT(model):
        return (model.xdash - model.model(model.observation_times, 
                                          *model.cs, *model.ps)
               ).reshape((-1,1))

    @staticmethod
    def _autoconfig_data(data: np.array, select: list=None) -> (dict, np.array):
        config = {'sz': data.shape}
        if not select:
            config['obs_fn'] = lambda x: (1-np.isnan(data).astype(float)) * x
        else:
            config['obs_fn'] = lambda x: ((1-np.isnan(data).astype(float)) * x)[:,select]
        return config

    @staticmethod
    def _autoconfig_L(data):
        config = {'n': np.prod(data.shape), 'diag': True}
        return config

    def make(self, config):
        assert len(config['L']) == len(config['Y'])
        for i, L in enumerate(config['L']):
            if not L['diag']:
                Lobj = ca.SX.sym(f'L_{i}', L['n'], L['n'])
                self.Ls.append(Lobj)
                self._Ls.append(Lobj)
            else:
                Lobj = ca.SX.sym(f'L_{i}')
                self.Ls.append(Lobj)
                self._Ls.append(ca.SX.eye(L['n'])*Lobj)
        for i, Y in enumerate(config['Y']):
            self.y0s.append(ca.SX.sym(f'Y0_{i}', *Y['sz']))
            self.ys.append(Y['obs_fn'])
        self.objective_function = sum(ca.sumsqr(L@(y0-y))/L.shape[0]
                                      for L, y0, y in zip(self._Ls, self.y0s, self.ys))
