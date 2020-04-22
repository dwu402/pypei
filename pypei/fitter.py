""" Interface to CasADi IPOPT interface and related UQ tools """
import numpy as np
import casadi as ca
from .functions import misc

class Solver():
    """ Solver interface to CasADi non linear solver """
    def __init__(self, config=None):
        self.solver = None
        self.objective_function = None
        self.constraints = None
        self.decision_vars = None
        self.parameters = None

        self._p_former = None

        self.__default_solve_opts__ = {
            'ipopt': {
                # standard verbosity
                'print_level': 5,
                # print every 50 iterations
                'print_frequency_iter': 50,
            }
        }
        self.solve_opts = self.__default_solve_opts__

        self.profilers = []

        if config:
            self.make(config)

    def make(self, config):
        """ Creates the solver

        Config Options
        --------------
        x, Decision Variables object
        f, Objective Function object
        g, Constraints object
        p, Parameters object (Fixed symbols that are not dependent on x)
        o, Options (see casadi.nlpsol) passed onto the IPOPT solver
        """
        self.decision_vars = config['x']
        self.parameters = config['p']
        self.objective_function = config['f']
        self.constraints = config['g']

        if 'o' in config:
            self.solve_opts = config['o']

        self.solver = ca.nlpsol(
            'solver', 'ipopt',
            {
                'x': self.decision_vars,
                'f': self.objective_function,
                'g': self.constraints,
                'p': self.parameters,
            },
            self.solve_opts
        )

    def __call__(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    @staticmethod
    def make_config(model, objective):
        """ Generates the default solver configuration

        x, Decision variables: [c, p] which are the spline coefficients and model parameters
        f, Objective Function: from objective object
        g, Constraints: On the state variables (e.g. for non-negativity)
        p, Parameters: L matrices and data
        """
        return {
            'x': ca.vcat([*model.cs, *model.ps]),
            'f': objective.objective_function,
            'g': model.xs.reshape((-1, 1)),
            'p': ca.vcat(misc.flat_squash(*objective.Ls, *objective.y0s))
        }

    def prep_p_former(self, objective):
        """ Create function to combine L and y0 for solver """
        self._p_former = ca.Function('p_former', objective.Ls + objective.y0s,
                                     [ca.vcat(misc.flat_squash(*objective.Ls, *objective.y0s))])

    def form_p(self, Ls, y0s):
        """ Combines inputs for L matrices and data for use in the solver """
        return self._p_former(*Ls, *y0s)

    @staticmethod
    def proto_x0(model):
        """ Generates initial iterates for the decision variables x = [c, p]

        This returns all ones of the correct shape, which can be further manipulated.
        """
        return {
            'x0': np.ones(ca.vcat([*model.cs, *model.ps]).shape),
            'c0': np.ones(ca.vcat(model.cs).shape),
            'p0': np.ones(ca.vcat(model.ps).shape)
        }

    def _profiler_configs(self, model):
        """ Default profiler configurations

        Profiling over all parameters individually
        """
        return [{'g+': p, 'pidx': ca.Function('pidx', [self.decision_vars], [p])} for p in model.ps]

    def make_profilers(self, configs):
        """ Creates profilers from configs

        Inherit problem structure from solver.

        Config options
        --------------
        g+: (required) symbolic that represents the parameter/expression being profiled
        pidx: function that determines the profiled expression's value in the MLE. Used by default bounds
        """
        for config in configs:
            self.profilers.append(Profiler(self, config))

    def profile(self, mle, p=None, lbx=-np.inf, ubx=np.inf, lbg=-np.inf, ubg=np.inf, pbounds=None):
        """ Executes the profilers

        Inherits the problem structure from solver.

        Parameters
        ----------
        mle: (dict) maximum likelihood estimate object. Output from solver run
        p: (dict) Parameter dict used for solver
        lbx: Lower bound input for solver
        ubx: Upper bound input for solver
        lbg: Lower bound on constraints used in solve of mle
        ubg: Upper bound on constraints used in solve of mle
        pbounds: (list) bounds on profling. Will default based on mle if not provided
        """
        profiles = []
        # create default bounds if none at all given
        if not pbounds:
            pbounds = [profiler._default_bound_range(mle) for profiler in self.profilers]
        for profiler, bound_range in zip(self.profilers, pbounds):
            profile = []
            # use default bounds if not given
            if bound_range is None:
                bound_range = profiler._default_bound_range(mle)
            for prfl_p in bound_range:
                plbg, pubg = profiler.set_g(profiler, prfl_p, lbg_v=lbg, ubg_v=ubg)
                profile.append(profiler.profiler(x0=mle['x'], p=p, lbx=lbx, ubx=ubx, lbg=plbg, ubg=pubg))
            profiles.append(profile)
        return profiles

    def get_parameters(self, solution, model):
        return ca.Function('pf', [self.decision_vars], model.ps)(solution['x'])

    def get_state(self, solution, model):
        return ca.Function('xf', [self.decision_vars], [model.xs])(solution['x'])

class Profiler():
    """ Tightly bound sub-object of Solver """
    def __init__(self, solver, config):
        profile_constraint = ca.vcat([solver.constraints, config['g+']])
        self.p_locator = config['pidx']
        self.profiler = ca.nlpsol('solver', 'ipopt',
                                  {
                                      'x': solver.decision_vars,
                                      'f': solver.objective_function,
                                      'g': profile_constraint,
                                      'p': solver.parameters,
                                  },
                                  solver.solve_opts)

    def set_g(self, bnd_value, lbg_v=-np.inf, ubg_v=np.inf):
        """ Creates the constraint bounds from existing solver bounds """
        # exploiting structure of Casadi.IpoptInterface
        gsz = self.profiler.size_in(2)
        lbg = ca.SX.ones(gsz)
        ubg = ca.SX.ones(gsz)
        lbg[:-1] = lbg_v
        ubg[:-1] = ubg_v
        lbg[-1] = bnd_value
        ubg[-1] = bnd_value
        return lbg, ubg

    def _default_bound_range(self, mle, num=20):
        mle_pval = self.p_locator(mle['x'])
        return np.linspace(0.5*mle_pval, 1.5*mle_pval, num=num, dtype=float)
