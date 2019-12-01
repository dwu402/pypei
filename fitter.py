import casadi as ca
import numpy as np
import pickle
import modeller
import copy
from multiprocessing import Pool

def argsplit(arg, n):
    """ Used for splitting the values of c into 3 c vectors for the model """
    try:
        assert len(arg)%n == 0
    except Exception as E:
        print(len(arg))
        raise E
    delims = [int(i*len(arg)/n) for i in range(n)] + [len(arg)]
    return [arg[delims[i]:delims[i+1]] for i in range(n)]

def tokey(root, branches):
    """ rho/p hasher """
    return f"{'y'.join(map(str, branches))}r{root}"

class Objective():
    """Object that represents the objective function to minimize

    This represents:
    J(theta, c) = ||w*p*(y-H*Phi*c)||^2 + lambda*||D(Phi*c) - f(Phi*c, theta)||^2
    where:
      w = weightings on state (diagonal)
      p = data density (diagonal)
      y = stacked data vector
      H = collocation matrix/observation model
      Phi = spline basis
      c = spline coefficients
      D = differential operator
      f = process model
      theta = process parameters
    """

    def __init__(self):
        self.m = 0
        self.observations = None
        self.collocation_matrices = None
        self.observation_vector = None
        self.weightings = None
        self.densities = None
        self.regularisation_vector = 0
        self.input_list = []

        self.rho = ca.MX.sym('rho')
        self.alpha = ca.MX.sym('alpha')

        self.obj_1 = None
        self.obj_2 = None
        self.regularisation = None
        self.objective = None

        self.obj_fn_1 = None
        self.obj_fn_2 = None
        self.reg_fn = None
        self.obj_fn = None

    def make(self, config, dataset, model):
        """Create the objective function"""

        self.m = len(dataset['t'])

        self.observation_vector = np.array(config['observation_vector'])
        self.weightings = np.array(config['weightings'][0])
        self.densities = np.array(config['weightings'][1])
        self.regularisation_vector = np.array(config['regularisation_value'])

        self.observations = self.observations_from_pandas(dataset['y'])
        self.collocation_matrices = self.colloc_matrices(dataset, model)
        
        self.input_list = [*model.cs, *model.ps]
        self.objective_input_list = [*self.input_list, self.rho, self.alpha]

        self.create_objective(model)
        self.create_objective_functions()

    def create_objective(self, model):
        self.obj_1 = sum(w/len(ov) * ca.sumsqr(self.densities*(ov - (cm@model.cs[j])))
                         for j, ov, w, cm in zip(self.observation_vector,
                                                 self.observations,
                                                 self.weightings,
                                                 self.collocation_matrices))
        self.obj_2 = sum(ca.sumsqr((model.xdash[:, i] -
                                      model.model(model.observation_times, *model.cs, *model.ps)[:, i]))
                          for i in range(model.s))/model.n

        self.regularisation = ca.sumsqr(ca.vcat(model.ps) - ca.vcat(self.regularisation_vector))

        self.objective = self.obj_1 + self.rho*self.obj_2 + self.alpha*self.regularisation

    def create_objective_functions(self):
        self.obj_fn_1 = ca.Function('fn1', self.input_list, [self.obj_1])
        self.obj_fn_2 = ca.Function('fn2', self.input_list, [self.obj_2])
        self.reg_fn = ca.Function('fn3', self.input_list, [self.regularisation])
        self.obj_fn = ca.Function('objective', self.objective_input_list, [self.objective])

    def observations_from_pandas(self, observations, convert=True):
        """Transposes pandas array to numpy array"""
        arr = np.stack(np.array(observations)).T
        for arr_row in arr:
            if len(arr_row) < self.m:
                arr = np.pad(arr, ((0, 0), (0, self.m-len(arr[0]))), 'constant', constant_values=0)
        if convert:
            arr = np.nan_to_num(arr, copy=True)
        return arr

    def colloc_matrices(self, dataset, model):
        """ Generate the matrix that represents the observation model, g

        This is a matrix, where the time points are mapped onto the finer time grid"""

        colloc_matrix_numerical = [np.zeros((self.m, model.K)) for i in range(dataset['y'].shape[-1])]
        for k, y in enumerate(dataset['y'].T): 
            for i, d_t in enumerate(dataset['t']):
                if not np.isnan(y[i]):
                    colloc_matrix_numerical[k][i, :] = [b(d_t) for b in model.basis_fns]

        return colloc_matrix_numerical

class SolveCache():
    def __init__(self):
        pass

    def new(self, key, value):
        pass

class InnerSolver(ca.Callback):
    def __init__(self, name, opts={}):
        ca.Callback.__init__(self)
        self.construct(name, opts)
    
    def init(self):
        self.cache = SolveCache()

class OuterGradient(ca.Callback):
    def __init__(self, name, opts={}):
        ca.Callback.__init__(self)
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 2
    def get_n_out(self): return 2

    def eval(self, x, c):
        pass

class Solver():
    """Encapsulation for the nlpsol calls to IPOPT

    Generates an inner and outer function to perform two stage profiling

    Uses the objective function MX object wrapped in Objective
    Assume that the problem is to minimize F with respect to x, with nuisance parameters c

    Inner objective calls (solve c wrt x):
    inner_problem = dict(x=c, p=x, f=F)
    c_min = ca.nlpsol('inner_objective', 'ipopt', inner_problem)['x']

    Outer Objective Gradient Mathematically:
    J = dF/dx - dF/dc * (d2F/dc2)^(-1) * (d2F/dxdc) [by implicit function theorem]

    Outer objective gradient construction:
    J = ca.gradient(F, x) - ca.gradient(F, c)/ca.gradient(ca.gradient(F, c), c)*ca.gradient(ca.gradient(F,c), x)
    temp = ca.MX.sym('temp')
    Jf = ca.Function('outer_gradient', [x, temp], [J])
    Jfn = OuterGradient(x, temp, F, Jf)

    Outer objective calls (solve x with optimal c)
    Fout = ca.substitute(F, c, c_min(x0=c))
    outer_problem = dict(x=x, f=Fout)
    options = dict(no_nlp_grad=True, grad_f=Jfn, verbose_init=True, 
                   ipopt=dict(hessian_approximation='limited-memory'))
    x_min = ca.nlpsol('outer_objective', 'ipopt', outer_problem, options)
    """

    def __init__(self, context):
        self.inner_objectives = []
        self.outer_objectives = []
        self.outer_gradients = []


class DirectSolver():
    """Directly optimises the objective function.
    Does not do the profiling step
    This is more computationally efficient, since it does not need to 
    make multiple calls to the IPOPT solver for a single theta, but may 
    run into local optima issues"""
    def __init__(self, config=None):
        self.models = []
        self.objectives = []
        self.solvers = []
        self.solve_opts = []
        self.solutions = []
        self.process_pool = None
        self.__util_p = None

        if config:
            self.build_models(config)
            self.build_objectives(config)
            self.build_solvers(config)
            self.prep_pool()

    def build_models(self, config):
        base_config = copy.copy(config.modelling_configuration)
        base_config['model'] = config.model
        for dataset, timespan in zip(config.datasets, config.time_span):
            model_config = base_config
            model_config['dataset'] = dataset
            model_config['time_span'] = timespan
            self.models.append(modeller.Model(model_config))
    
    def build_objectives(self, config):
        for model, dataset in zip(self.models, config.datasets):
            objective = Objective()
            objective.make(config.fitting_configuration, dataset, model)
            self.objectives.append(objective)

    def build_solvers(self, config):
        for i, (objective, model) in enumerate(zip(self.objectives, self.models)):
            problem = {
                'f': objective.objective,
                'x': ca.vcat(objective.input_list),
                'p': ca.vcat([objective.rho, objective.alpha]),
                'g': ca.vcat(model.ps),
            }
            options = {
                'ipopt': {
                    'print_level': 3,
                },
            }
            self.solvers.append(ca.nlpsol(f"solver{i}", 'ipopt', problem, options))
            opts = self.prep_solver(config, model)
            self.solve_opts.append((i, opts))

    def prep_solver(self, config, model):
        c0 = np.ones(model.K*model.s)
        return {
            'x0' : np.concatenate([c0, config.initial_parameters]),
        }

    def solve_problem(self, inputs):
        idx, opts = inputs
        res = self.solvers[idx](**opts)
        return np.array(res['x'])

    def prep_pool(self, ncpus=2):
        self.process_pool = Pool(ncpus)

    def solve_problems(self, custom_opts, propagate=False):
        # reconstruct with custom options

        solve_opts = copy.copy(self.solve_opts)
        for (idx, opt) in solve_opts:
            opt.update(custom_opts)

        return self.process_pool.map(self.solve_problem, solve_opts)

    def getp(self, idx, solution):
        if self.__util_p is None:
            self.__util_p = ca.Function('getp', 
                                        ca.vcat(self.objectives[idx].input_list), 
                                        self.models[idx].ps)
        return self.__util_p(solution['x'])