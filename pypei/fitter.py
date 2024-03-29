""" Interface to CasADi IPOPT interface and related UQ tools """
import itertools
import numpy as np
import casadi as ca
from scipy import stats
from .functions import misc

ipopt_reduced = {
    'ipopt': {
        # standard verbosity
        'print_level': 5,
        # print every 50 iterations
        'print_frequency_iter': 50,
        # for correct multipliers on fixed decision variables?
        # 'fixed_variable_treatment': 'make_constraint',
        # diagnostic strings
        'print_info_string': 'yes',
    }
}

ipopt_silent = {
    # No output except initial banner
    'ipopt': {
        'print_level': 0,
    },
    'print_time': 0
}

class Solver():
    """ Solver interface to CasADi non linear solver """
    def __init__(self, config=None):
        self.solver = None
        self.objective_function = None
        self.constraints = None
        self.decision_vars = None
        self.parameters = None

        self._solver = None

        self._p_former = None

        self.__default_solve_opts = ipopt_reduced

        self.solve_opts = self.__default_solve_opts

        self.profilers = []

        self.__evaluator = None

        if config:
            self.make(config)

    def __str__(self):
        dv_sz = getattr(self.decision_vars, 'shape', None)
        p_sz = getattr(self.parameters, 'shape', None)
        g_sz = getattr(self.constraints, 'shape', None)
        of_sz = getattr(self.objective_function, 'shape', None)
        return f"pypei Solver wrt {dv_sz} s/t {g_sz} params {p_sz}"

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

        self.construct()
        if self._solver is None:
            self._solver = self.solver

    def construct(self):
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

    def eval_at(self, x, p):
        if not self.__evaluator:
            self.__evaluator = ca.Function('evaluator', 
                                           [self.decision_vars, self.parameters], 
                                           [self.objective_function])
        return self.__evaluator(x, p)

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

    def add_profilers(self, configs):
        """ Creates profilers from configs

        Inherit problem structure from solver.

        Config options
        --------------
        g+: (required) symbolic that represents the parameter/expression being profiled
        pidx: function that determines the profiled expression's value in the MLE. Used by default bounds
        """
        for config in configs:
            self.profilers.append(Profiler(self, config))


    def make_profilers(self, configs):
        """ Creates profilers from configs.
        Clears existing profilers

        Inherit problem structure from solver.

        Config options
        --------------
        g+: (required) symbolic that represents the parameter/expression being profiled
        pidx: function that determines the profiled expression's value in the MLE. Used by default bounds
        """
        self.profilers = []
        self.add_profilers(configs)

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
        self.constraint_var = config['g+']
        profile_constraint = ca.vcat([solver.constraints, config['g+']])
        self.nnz = config['g+'].nnz()
        self.p_locator = config.get('pidx', None)
        self.profiler = ca.nlpsol('solver', 'ipopt',
                                  {
                                      'x': solver.decision_vars,
                                      'f': solver.objective_function,
                                      'g': profile_constraint,
                                      'p': solver.parameters,
                                  },
                                  solver.solve_opts,
                                 )
        
    def __call__(self, *args, **kwargs):
        return self.profiler(*args, **kwargs)

    def set_g(self, bnd_value, lbg_v=-np.inf, ubg_v=np.inf):
        """ Creates the constraint bounds from existing solver bounds """
        gsz = self.profiler.size_in(4)
        lbg = np.ones(gsz).flatten()
        ubg = np.ones(gsz).flatten()
        lbg[:-self.nnz] = lbg_v
        ubg[:-self.nnz] = ubg_v
        lbg[-self.nnz:] = bnd_value
        ubg[-self.nnz:] = bnd_value
        return lbg, ubg

    def _default_bound_range(self, mle, num=21, variance=0.5):
        """ Generates a range of bounds from (1-v)*m to (1+v)*m 

        where v is the variance parameter
        and m is the value of the profiler constraint at the MLE

        Works for 1D only.
        """
        mle_pval = self.p_locator(mle['x'])
        return np.linspace((1-variance)*mle_pval, (1+variance)*mle_pval, num=num, dtype=float).flatten()

    def simple_nvariate_bound_sets(self, mle, num=21, variance=0.5):
        """  Generates a range of bounds from (1-v)*m to (1+v)*m 

        where v is the variance parameter
        and m is the value of the profiler constraint at the MLE
        """
        mle_pval = self.p_locator(mle['x'])
        bound_range = np.linspace((1-variance)*mle_pval, (1+variance)*mle_pval, num)
        bound_meshgrids = np.meshgrid(*bound_range.T)
        bound_xs = list(zip(*[mesh.flatten() for mesh in bound_meshgrids]))
        return bound_xs

    def symmetric_bound_sets(self, mle, num=21, variance=0.5):
        """ Generates two bound ranges which are symmetric about the MLE value

        They start at the MLE and go towards the extremes of (1-v)*m and (1+v)*m
        where v is the variance parameter
        and m is the value of the profiler constraint at the MLE

        Works for 1D only.
        """
        mle_pval = self.p_locator(mle['x'])
        n = num//2 + 1
        return [np.linspace(mle_pval, (1-variance)*mle_pval, num=n, dtype=float).flatten(), 
                np.linspace(mle_pval, (1+variance)*mle_pval, num=n, dtype=float).flatten(),]

    def symmetric_nvariate_bound_sets(self, mle, num=21, variance=0.5):
        """ Generates four bound ranges which are symmetric about the MLE value

        They start at the MLE and traverse each quadrant in a snake-like manner.
        Curently only does a square grid.
        """
        mle_pval = self.p_locator(mle['x']).toarray().flatten()
        n = num//2 + 1

        mle_hilo = [((x, 1), (x, -1)) for x in mle_pval]

        bound_sets = []
        for (xm, xv), (ym, yv) in itertools.product(*mle_hilo):
            x_arr = np.linspace(xm, xm*(1+ xv*variance), n)
            y_arr = np.linspace(ym, ym*(1+ yv*variance), n)
            Xgrid, Ygrid = np.meshgrid(x_arr, y_arr)
            bound_sets.append(list(zip(self.diag_mat(Xgrid), self.diag_mat(Ygrid))))
        return bound_sets

    @staticmethod
    def is_nonmonotone_points(seq, reverse=False):
        """ Returns a boolean array that takes value TRUE if the point is nonmonotonic """
        return [reverse ^ any(x > xi for xi in seq[i+1:]) for i,x in enumerate(seq)]

    @staticmethod
    def resolve_seqs(ll_seq):
        """ Returns root:subseq map of subsequences should be re-solved from root 
        (due to monotonicity)"""
        grps = map(lambda x:(x[0][0], len(x)), [list(l) for k,l in itertools.groupby(enumerate(ll_seq), key=lambda x:x[1]) if k])
        return {(i+l): slice(i+l-1, i-1, -1) for i, l in grps}

    @staticmethod
    def concave_up(seq):
        """ Returns whether inner points of a sequence are concave up 
        Outer points automatically resolve to True"""
        return [True] + [float(f1/2 - f2 + f3/2) > 0 for f1,f2,f3 in zip(seq[:-2], seq[1:-1], seq[2:])] + [True]

    @staticmethod
    def diag_mat(mat, n=None):
        """Diagoanlly traverses through a matrix
        
        Adapted from an implementatation found on geeksforgeeks
        """
        if n is None:
            n = mat.shape[0]
        mode = 0
        it = 0
        lower = 0

        # 2n will be the number of iterations
        for t in range(2 * n - 1):
            t1 = t
            if (t1 >= n):
                mode += 1
                t1 = n - 1
                it -= 1
                lower += 1
            else:
                lower = 0
                it += 1

            for i in range(t1, lower - 1, -1):
                if ((t1 + mode) % 2 == 0):
                    yield mat[i, t1 + lower - i]
                else:
                    yield mat[t1 + lower - i, i]


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
    resamples = [[stats.norm(mu, np.sqrt(var)).rvs(random_state=None) 
                  for mu, var in zip(mle_y, variances)] for _ in range(num)]

    return resamples