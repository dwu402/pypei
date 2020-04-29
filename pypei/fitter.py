""" Interface to CasADi IPOPT interface and related UQ tools """
import numpy as np
import casadi as ca
from scipy import stats
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
                plbg, pubg = profiler.set_g(prfl_p, lbg_v=lbg, ubg_v=ubg)
                profile.append(profiler.profiler(x0=mle['x'], p=p, lbx=lbx, ubx=ubx, lbg=plbg, ubg=pubg))
            profiles.append({'ps': bound_range, 'pf': profile})
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
        gsz = self.profiler.size_in(4)
        lbg = np.ones(gsz)
        ubg = np.ones(gsz)
        lbg[:-1] = lbg_v
        ubg[:-1] = ubg_v
        lbg[-1] = bnd_value
        ubg[-1] = bnd_value
        return lbg, ubg

    def _default_bound_range(self, mle, num=20):
        mle_pval = self.p_locator(mle['x'])
        return np.linspace(0.5*mle_pval, 1.5*mle_pval, num=num, dtype=float).flatten()


"""
Resampling Tools
"""
def reconfig_rto(model, objective, solver, config, index=None):
    """ Reconfigure unitary data in the objective for full resampling

    Parameters
    ----------
    solver: Solver object to rebuild
    objective: Objective object to rebuild
    config: Original objective configuration
    """

    # expand objective
    if index is None:
        for i, c in enumerate(config['Y']):
            if 'unitary' in c and c['unitary']:
                y0 = ca.SX.sym(f'Y0_{i}', *c['sz'])
                objective.y0s[i] = y0.reshape((-1, 1))
                objective._y0s[i] = y0.reshape((-1, 1))
    else:
        y0 = ca.SX.sym(f'Y0_{index}', *config['Y'][index]['sz'])
        objective.y0s[index] = y0.reshape((-1, 1))
        objective._y0s[index] = y0.reshape((-1, 1))
    objective.assemble_objective()

    # rebuild solver
    s_config = solver.make_config(model, objective)
    solver.make(s_config)
    solver.prep_p_former(objective) 

def get_mle_y(objective, solver, mle):
    x2y = ca.Function('x2y', [solver.decision_vars], objective.ys)
    return x2y(mle['x'])

def var_from_mle_and_data(mle_y, data):
    return [((y-x).T@(y-x))/(x.numel()-1) for y, x in zip(data, mle_y)]

def estimate_variances(objective, solver, mle, data):
    mle_y = get_mle_y(objective, solver, mle)
    return var_from_mle_and_data(mle_y, data)

def gaussian_resampling(objective, solver, mle, data, num=50):
    mle_y = get_mle_y(objective, solver, mle)
    variances = var_from_mle_and_data(mle_y, data)
    resamples = [[stats.norm(mu, np.sqrt(var)).rvs(random_state=None) for mu, var in zip(mle_y, variances)] for _ in range(num)]

    return resamples
