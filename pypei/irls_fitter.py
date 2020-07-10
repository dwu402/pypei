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

import casadi as ca
from numpy import sqrt
from . import fitter
from .functions.misc import func_kw_filter

@func_kw_filter
def _gaussian_weight_function(residuals, n_obsv):
    """ Gaussian weights for pypei.Objective 
    We know that the weights are 1/sigma and that sigma^2 = f/n
    """
    # TODO: determine if n_obsv is available automatically
    return 1/sqrt([float(r)/n_obsv for r in residuals])

_known_weight_functions = {
    "gaussian": _gaussian_weight_function
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
        def solver(*args, **kwargs):
            return self._solver(*args, **{k:v for k,v in kwargs.items() 
                                          if k in self._solver.name_in()})
        self.solver = solver

    def irls(self, x0, p, w0=None, nit=4, weight="gaussian", hist=False, **kwargs):
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
        **kwargs : 
            Additional keyword arguments passed to the solver and weighting functions
        """

        assert nit >= 1, f"Number of iterations specified <{nit}> is less than 1."
        if isinstance(weight, str):
            assert weight in _known_weight_functions, \
                f"Weight function type <{weight}> not understood."
            weight_fn = _known_weight_functions[weight]
        else:
            weight_fn = func_kw_filter(weight)
        
        if w0 is None:
            weights = [1] * len(self.objective_obj.Ls)
        else:
            weights = w0

        if hist:
            mu_hist = []
            w_hist = [weights]

        # from knowing the general form of the objective function, we can deduce
        # that the weights will be tied to the unweighted components (residuals)
        residual_fn = ca.Function("mu",
                                  [self.decision_vars, self.parameters],
                                  [self.objective_obj.us_obj_fn(i)
                                   for i in range(len(self.objective_obj.ys))])

        for _ in range(nit):
            p_of_ws = p(weights)
            sol = self.solver(x0=x0, p=p_of_ws, **kwargs)
            x0 = sol['x'].toarray().flatten()
            residuals = residual_fn(sol['x'], p_of_ws)
            weights = weight_fn(residuals=residuals, **kwargs)
            if hist:
                mu_hist.append(sol)
                w_hist.append(weights)

        if hist:
            return sol, weights, mu_hist, w_hist

        return sol, weights

    def profile(self, *args, **kwargs):
        # TODO: Construct an equivalent profiling setup
        return super().profile(*args, **kwargs)