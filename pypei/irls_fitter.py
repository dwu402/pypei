""" New solve scheme: iteratively reweighted least squares

initialise w, x0

for i in 1..N; do
    mu <- solve weighted least squares problem with
            weights w and initial iterate x0
    w <- Update weights based on mu
            For Gaussian noise, this reduces to
            computing the sample variance
    x0 <- Update the initial iterate from mu

"""

import warnings
import casadi as ca
from functools import wraps
from numpy import array, sqrt, inf, zeros
from numpy.random import default_rng
from . import fitter
from .functions.misc import func_kw_filter

random = default_rng()

@func_kw_filter
def _gaussian_weight_function(residuals, n_obsv):
    """ Gaussian weights for pypei.Objective 
    We know that the weights are 1/sigma and that sigma^2 = f/n
    """
    # TODO: determine if n_obsv is available automatically
    return 1/sqrt([float(ca.sumsqr(r))/n_obsv for r in residuals])

@func_kw_filter
def _gaussian_inverse_weight_function(weights):
    """ Mapping from weights to variance """
    return 1/array(weights)**2

_known_weight_functions = {
    "gaussian": _gaussian_weight_function
}

_inverse_weight_functions = {
    'gaussian': _gaussian_inverse_weight_function
}

class Solver(fitter.Solver):
    """Iteratively Reweighted Least Squares on top of Casadi nlpsol interface
    
    The IRLS algorithm allows estimation of non-iid or non-normal covariance
    structures. It works by iteratively updating the weights that are used in 
    the computation of the objective based on the previous estimate.

    In the pypei objective strucutre, these weights correspond to the inverse of
    the square root of the determinant of the variance.
    """
    def __init__(self, objective=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # need to assign later if not given here, used for irls algorithm weights
        self.objective_obj = objective
        self.residual_function = None

    def __call__(self, *args, **kwargs):
        self.irls(*args, **kwargs)

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

        super().make(config)
        # keyword filter the solver
        self._solver = self.solver
        @wraps(self._solver)
        def solver(*args, **kwargs):
            return self._solver(*args, **{k:v for k,v in kwargs.items() 
                                          if k in self._solver.name_in()})
        self.solver = solver
        # from knowing the general form of the objective function, we can deduce
        # that the weights will be tied to the unweighted components (residuals)
        self.residual_function = ca.Function("mu",
                                             [self.decision_vars, self.parameters],
                                             [self.objective_obj.us_obj_comp(i)
                                              for i in range(len(self.objective_obj.ys))])

    def irls(self, x0, p, w0=None, nit=4, weight="gaussian", hist=False, solver=None, weight_args=None, **solver_args):
        """ Performs iteratively reweighted least squares

        Parameters
        ----------
        x0 : 
            Initial iterate for the decision variables.
        p : callable
            Function that, given the weights, returns the solver parameters
        w0 :
            Initial iterate for the weights. Defaults to a ones-like
        nit : int
            Number of iterations.
            Acts as a regularisation hyperparameter
        weight : str or callable
            A string from the list of known weight functions or a callable
            that returns the updated weights, when given the unweighted 
            residuals from the current solution. 
            Defaults to (component-wise) Gaussian: weights = 1/(sqrt(f/n)))
        hist : bool
            Whether or not to record the history of decision variable solutions
            and weights
        solver : Casadi.nlpsol object or None
            Object to solve with. If None, defaults to self.solver
        weight_args : dict
            Additional information passed to the weight function as arguments
        solver_args : 
            Additional keyword arguments passed to the solver
        """

        assert nit >= 1, f"Number of iterations specified <{nit}> is less than 1."
        if isinstance(weight, str):
            assert weight in _known_weight_functions, \
                f"Weight function type <{weight}> not understood."
            weight_fn = _known_weight_functions[weight]
        else:
            weight_fn = weight
        
        if w0 is None:
            weights = [1] * len(self.objective_obj.Ls)
        else:
            weights = w0

        if hist:
            mu_hist = []
            w_hist = [weights]

        if solver is None:
            solver = self.solver

        if weight_args is None:
            weight_args = {}

        for _ in range(nit):
            p_of_ws = p(weights)
            sol = solver(x0=x0, p=p_of_ws, **solver_args)
            x0 = sol['x'].toarray().flatten()
            residuals = self.residual_function(sol['x'], p_of_ws)
            weights = weight_fn(residuals=residuals, **weight_args)
            if hist:
                mu_hist.append(sol)
                w_hist.append(weights)

        if hist:
            return sol, weights, mu_hist, w_hist

        return sol, weights

    def profile(self, mle, p=None, w0=None, nit=4, weight="gaussian", lbx=-inf, ubx=inf, lbg=-inf, ubg=inf, pbounds=None, weight_args=None, **kwargs):
        # TODO: Construct an equivalent profiling setup
        profiles = []
        if not pbounds:
            pbounds = [profiler._default_bound_range(mle) for profiler in self.profilers]
        for profiler, bound_range in zip(self.profilers, pbounds):
            profile = {'s': [], 'w': []}
            if bound_range is None:
                bound_range = profiler._default_bound_range(mle)
            for profile_p in bound_range:
                plbg, pubg = profiler.set_g(profile_p, lbg_v=lbg, ubg_v=ubg)
                s, w = self.irls(mle['x'], p=p, w0=w0, nit=nit, weight=weight, lbx=lbx, ubx=ubx, lbg=plbg.flatten(), ubg=pubg.flatten(), hist=False, solver=profiler, weight_args=weight_args, **kwargs)
                profile['s'].append(s)
                profile['w'].append(w)
            profiles.append({'ps': bound_range, 'pf': profile})
        return profiles

    def gaussian_resample(self, mle, data, ws, objective, nsamples, reconfigure=False, **kwargs):
        if reconfigure:
            warnings.warn("RTO reconfiguration is best done outside of the resampling function", RuntimeWarning)
            assert 'model' in kwargs, "Model not provided"
            assert 'config' in kwargs, "Objective Configuration not provided"
            index = kwargs['index'] if 'index' in kwargs else None
            fitter.reconfig_rto(kwargs['model'], objective, self, kwargs['config'], index=index)

        # construct the y0s to refit
        # modelling y0 ~ y + N(0, G)
        resampled_y0s = []
        for y, L in zip(data, objective._Ls):
            # compute G = (L.T)^-1((L.T)^-1).T
            # extract L via p and objective.Ls
            G = ca.Function('w2G', [*objective.Ls, self.decision_vars], [ca.inv(L.T)@(ca.inv(L.T).T)])(*ws, mle['x'])
            mean = zeros(L.size(1))
            samples = random.multivariate_normal(mean, G, size=nsamples)
            # y_base = ca.Function('yfromdv', [self.decision_vars], [y])(mle['x'])
            resampled_y0s.append(samples + y)

        resample_sols = []
        for sample in zip(*resampled_y0s):
            # construct the p function to pass to irls
            def w2p(w):
                return [*w, *[i for j in sample for i in j]]
            resample_sols.append(self.irls(mle['x'], p=w2p, hist=False, **kwargs))

        return resample_sols, resampled_y0s
