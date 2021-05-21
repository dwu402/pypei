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
from numpy import array, sqrt, inf, zeros, abs
from numpy.linalg import norm as npnorm, solve as linsolve
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
    return 1/sqrt([float(ca.sumsqr(r))/n for r,n in zip(residuals, n_obsv)])

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
        self.residual = ca.Function("residual",
                                    [self.decision_vars, self.parameters],
                                    [self.objective_obj.log_likelihood])
        self.component_residuals = ca.Function("mu",
                                               [self.decision_vars, self.parameters],
                                               [self.objective_obj.us_obj_comp(i)
                                                for i in range(len(self.objective_obj.ys))])

    @staticmethod
    def _default_p(ws, data):
        return ca.vcat([ws, data])

    def irls(self, x0, p=None, y=None, w0=None, nit=4, weight="gaussian", hist=False, step_control=None, solver=None, weight_args=None, **solver_args):
        """ Performs iteratively reweighted least squares

        Parameters
        ----------
        x0 : 
            Initial iterate for the decision variables.
        p : callable
            Function that, given the weights (and data), returns the solver parameters
        y :
            Value to pass through to p (represents data)
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
        step_control : dict
            Dictionary of parameters to control step-correction
                maxiter: maximum number of correction steps to take in one iteration
                eps : threshold value for acceptable relative reduction in deviance
                gamma : weighting value for uniform test (relativve reduction)
        solver : Casadi.nlpsol object or None
            Object to solve with. If None, defaults to self.solver
        weight_args : dict
            Additional information passed to the weight function as arguments
        solver_args : 
            Additional keyword arguments passed to the solver
        """

        assert nit >= 1, f"Number of iterations specified <{nit}> is less than 1."
        
        if p is None:
            p = self._default_p
        
        if isinstance(weight, str):
            assert weight in _known_weight_functions, \
                f"Weight function type <{weight}> not understood."
            weight_fn = _known_weight_functions[weight]
        else:
            weight_fn = weight
        
        if w0 is None:
            weights = [1] * ca.vcat(self.objective_obj.Ls).numel()
        else:
            weights = w0

        if hist:
            raw_sol_hist = []
            sol_hist = []
            w_hist = [weights]
            ctrl_hist = []

        if step_control is None:
            # defaults taken from glm2.fit
            step_control = {
                'maxiter': 5,
                'eps': 1e-8,
                'gamma': 0.1
            }

        if solver is None:
            solver = self.solver

        if weight_args is None:
            weight_args = {}

        out_sol = None
        for i in range(nit):
            p_of_ws = p(weights, y)
            sol = solver(x0=x0, p=p_of_ws, **solver_args)
            if i > 0:
                try:
                    x0, residual, controls = self._irls_step_control(
                        sol['x'].toarray().flatten(),
                        lambda x: self.residual(x, p_of_ws),
                        x0, residual, step_control,
                    )
                    if controls[1] >= 1:
                        print("Step control adjusted", controls[1]+1, "times at iteration", i)
                    out_sol = sol
                except Solver.StepControlError as step_err:
                    print(step_err)
                    print("Early termination at iteration", i+1, "due to divergence of objective function")
                    break
            else:
                x0 = sol['x'].toarray().flatten()
                residual = float(self.residual(x0, p_of_ws))
                controls = (None, None)
            component_residuals = self.component_residuals(x0, p_of_ws)
            weights = weight_fn(residuals=component_residuals, **weight_args)
            if hist:
                raw_sol_hist.append(sol)
                sol_hist.append({'x': x0, 'f': residual})
                w_hist.append(weights)
                ctrl_hist.append(controls)

        if hist:
            return out_sol, weights, sol_hist, w_hist, raw_sol_hist, ctrl_hist

        return sol, weights

    class StepControlError(RuntimeError):
        pass

    @staticmethod
    def _irls_step_control(x0, residual_function, old_x, old_residual, controls):
        """ Step control for IRLS inspired by glm2.fit from R/CRAN
        """

        for i in range(controls['maxiter']):
            residual = float(residual_function(x0))
            err = (residual - old_residual)#/(controls['gamma'] + abs(residual))
            # print(residual, old_residual, err)
            if err < controls['eps']:
                break
            x0 = (x0 + old_x) / 2
        else:
            raise Solver.StepControlError(f"Step control did not converge after {i+1} iterations, minimum improvement gained was {err}.")
        return x0, residual, (err, i)

    def profile(self, mle, p=None, w0=None, nit=4, weight="gaussian", lbx=-inf, ubx=inf, lbg=-inf, ubg=inf, pbounds=None, weight_args=None, restart=False, **kwargs):
        profiles = []
        if not pbounds:
            pbounds = [profiler.symmetric_bound_sets(mle) for profiler in self.profilers]
        for profiler, bound_set in zip(self.profilers, pbounds):
            profile = {'s': [], 'w': []}
            if bound_set is None:
                bound_set = profiler.symmetric_bound_sets(mle)
            for bound_range in bound_set:
                init_x = mle['x']
                for profile_p in bound_range:
                    plbg, pubg = profiler.set_g(profile_p, lbg_v=lbg, ubg_v=ubg)
                    s, w = self.irls(init_x, p=p, w0=w0, nit=nit, weight=weight, lbx=lbx, ubx=ubx, lbg=plbg.flatten(), ubg=pubg.flatten(), hist=False, solver=profiler, weight_args=weight_args, **kwargs)
                    if not restart:
                        init_x = s['x']
                    profile['s'].append(s)
                    profile['w'].append(w)
            profiles.append({'ps': bound_set, 'pf': profile})
        return profiles

    def _generate_gaussian_samples(self, mle, p, data, ws, objective, nsamples):
        resampled_y0s = []
        for i, (y, L) in enumerate(zip(data, objective._Ls)):
            # sample gaussian with a given cholesky decomp of a precision matrix
            L_real = ca.Function(f'Lreal{i}', [ca.vcat(objective.Ls), self.decision_vars], [L])(ws, mle['x'])
            Z = random.standard_normal((L.size(1), nsamples))
            # modelling y0 ~ y + N(0, G)
            X = linsolve(L_real, Z)

            resampled_y0s.append(X.T + y)
        
        return resampled_y0s

    def _fit_samples(self, samples, x0, p, w0, **kwargs):
        try:
            i = -1
            resample_sols = []
            for i, sample in enumerate(zip(*samples)):
                print("Fitting Sample", i)
                resample_sols.append(self.irls(x0, p=p, y=sample, w0=w0, **kwargs))
        except KeyboardInterrupt:
            print("Stopped at iteration", i)
            return resample_sols
        return resample_sols

    def gaussian_resample(self, mle, p, data, ws, objective, nsamples, reconfigure=False, **kwargs):
        """
        Inputs
        ------
        mle: Dictionary from nlpsol that contains the decision variable vector in 'x'
        p: Function that, given the weights, returns the solver parameters
        data: data that is fed to the sampler
        ws: weights that correspond with the MLE
        objective: pypei.Objective object
        nsamples: number of samples to draw
        reconfigure: recc. False, True if need to reconfigure the objective problem for RTO
        **kwargs: arguments passed to solver
        """
        if reconfigure:
            warnings.warn("RTO reconfiguration is best done outside of the resampling function", RuntimeWarning)
            assert 'model' in kwargs, "Model not provided"
            assert 'config' in kwargs, "Objective Configuration not provided"
            index = kwargs['index'] if 'index' in kwargs else None
            fitter.reconfig_rto(kwargs['model'], objective, self, kwargs['config'], index=index)

        # construct the y0s to refit
        resampled_y0s =  self._generate_gaussian_samples(mle, p, data, ws, objective, nsamples)

        resample_sols = self._fit_samples(resampled_y0s, mle['x'], p, ws, **kwargs)

        return resample_sols, resampled_y0s
