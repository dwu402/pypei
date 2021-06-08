from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca

from . import modeller, objective, fitter, irls_fitter

class Problem():
    def __init__(self):
        """An opinionated pypei Problem builder
        
        Infers parameters and estimates states of a specified ODE model from given data
        """
        self.model_function: Callable = None
        self.model_form: dict = None
        self.model_config: dict = None
        self.model: modeller.Model = None
        self.model_dt = None
        
        self.data_orig = None
        self.data = None
        self.data_time = None
        self.data_indexer = None
        self.interpolator = None

        self.objective_config: dict = None
        self.objective = objective.Objective()

        self.solver_config: dict = None
        self.solver: irls_fitter.Solver = None

        self.weight_fn: Callable = None

        self.initial_guess = None
        self.initial_weight = None
        self.lbx = None
        self.ubx = None
        self.lbg = None
        self.ubg = None    

    @staticmethod
    def p(w, y):
        return [*w, *y, 0]

    @staticmethod
    def gaussian_w(residual, n):
        return 1/np.sqrt(float(ca.sumsqr(residual))/n)

    def struct_weight_2(self, residuals):
        ws = []
        # deal with data weights
        for s in self.objective_config['L'][0]['struct']:
            ws.append(self.gaussian_w(residuals[0][s['i0']:s['i0']+s['n']], s['n']))
        # deal with model weights
        for s in self.objective_config['L'][1]['struct']:
            if 'i0s' in s:
                rs = ca.vcat([residuals[1][i0:i0+n] for n, i0 in zip(s['ns'], s['i0s'])])
                ws.append(self.gaussian_w(rs, sum(s['ns'])))
            else:
                ws.append(self.gaussian_w(residuals[1][s['i0']:s['i0']+s['n']], s['n']))
        return ws

    def build_model(self, model_fn, model_form, time_span, grid_size=200, basis_number=40):
        self.model_function = model_fn
        self.model_form = model_form
        self.model_config = {
            'grid_size': grid_size,
            'basis_number': basis_number,
            'model_form': model_form,
            'time_span': time_span,
            'model': model_fn,
        }
        self.model = modeller.Model(self.model_config)
        self.model_dt = np.gradient(self.model.observation_times)

    @staticmethod
    def slice_data(data_time, data, start=None, clip=None):
        return data_time[start:clip], data[:,start:clip]

    def build_data(self, data_time, data, ix=slice(0, None), iy=slice(0, None)):
        """ Build data related objects.

        Objective = f(data - interpolant)
        
        Parameters
        ----------
        data_time: Nx1 array of time points that data is observed at
        data: NxM array of data observed
        ix: x-direction slicer for interpolant object
        iy: y-direction slicer for interpolant object
        """
        self.data_time = data_time
        self.data_orig = data
        all_x = self.model.all_x_at(self.model.cs, data_time)[ix, iy]
        indexer, (data_filt, interpolator) = objective.ignore_nan(data, all_x.T)
        self.data_indexer = indexer
        self.data = data_filt
        self.interpolator = interpolator

    def build_objective(self, model_struct):
        """ Make an opinionated objective function 
        
        Does not regularise. Assumes form
        ||L(y-x)||^2 + ||W(dxdt - f(x,p))||^2

        Input is a dict for pypei.objective.map_order_to_L_struct
        """
        data_L = objective.Objective._autoconfig_L(self.data)
        objective.L_via_data(data_L, self.data_orig)
        data_L['balance'] = True
        model_L = {
            'n': np.prod(self.model.xs.shape),
            'iid': False,
            'balance': True,
            'struct': objective.map_order_to_L_struct(**model_struct, n_sz=self.model_config['grid_size']),
            'numL': len(model_struct['order']),
        }
        self.objective_config = {
            'Y':[
                {
                    'sz': self.data.shape,
                    'obs_fn': self.interpolator,
                },
                {
                    'sz': self.model.xs.shape,
                    'unitary': True,
                    'obs_fn': objective.Objective._MODELFIT(self.model),
                },
            ],
            'L':[
                data_L,
                model_L,
            ]
        }
        self.objective = objective.Objective()
        self.objective.make(self.objective_config)

    def build_solver(self, solver_opts=None, guess_opts=None, constraint_opts=None, w0=None):
        """ Build a Solver with opinionated initial guesses and error structures"""
        # make Solver object
        self.solver = irls_fitter.Solver(objective=self.objective)
        self.solver_config = self.solver.make_config(self.model, self.objective)
        if solver_opts is not None:
            self.solver_config['o'] = solver_opts
        self.solver.make(self.solver_config)

        # make opinionated guesses
        if guess_opts is None:
            guess_opts = dict()
        x0 = self.solver.proto_x0(self.model)
        x0x0 = x0['x0']
        n_ps = self.model_form['parameters']
        x0x0[:-n_ps] = np.random.poisson(x0x0[:-n_ps] * guess_opts.get('x0', 10_000))
        x0x0[-n_ps:] = x0x0[-n_ps:] * guess_opts.get('p0', 0.2)
        self.initial_guess = x0x0

        if constraint_opts is None:
            constraint_opts = dict()
        
        if 'lbx' not in constraint_opts:
            self.lbx = -np.inf*x0x0
            self.lbx[-n_ps:] = 0
        else:
            self.lbx = constraint_opts.get('lbx')
        self.ubx = constraint_opts.get('ubx', np.inf*x0x0)

        self.lbg = constraint_opts.get('lbg', 0)
        self.ubg = constraint_opts.get('ubg', 200_000)

        self.weight_fn = self.struct_weight_2

        if w0 is None:
            self.initial_weight = ([1] * self.objective_config['L'][0]['numL'] + 
                                  [2e-2] * self.objective_config['L'][1]['numL'])
        else:
            self.initial_weight = w0

    def solve(self, nit=6, hist=True):
        solution = self.solver.irls(self.initial_guess, p=self.p, y=self.data, nit=nit,
                                    lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg,
                                    w0=self.initial_weight, weight=self.weight_fn, hist=hist)

        return {k:v for k,v in zip(['sol', 'ws', 'shist', 'whist', 'raw_shist', 'ctrl_hist'], solution)}

    def plot_solution(self, solution, it=-1, ax=None, data=False):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot()
        ax.plot(self.model.observation_times, self.solver.get_state(solution['shist'][it], self.model))
        if data:
            ax.plot(self.data_time, self.data_orig.T, 'o')
