import copy
import casadi as ca
import numpy as np
from . import modeller

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
        self.observation_model = None
        self.observation_vector = None
        self.weightings = None
        self.densities = None
        self.regularisation_model = None
        self.regularisation_vector = 0
        self.input_list = []

        self.rho = ca.SX.sym('rho')
        self.alpha = ca.SX.sym('alpha')

        self.obj_1 = None
        self.obj_2 = None
        self.regularisation = None
        self.objective = None

        self.obj_fn_1 = None
        self.obj_fn_2 = None
        self.reg_fn = None
        self.obj_fn = None

    def make(self, config, model):
        """Parse inputs and create the objective function"""

        dataset = config['dataset']

        self.m = len(dataset['t'])

        if 'observation_model' in config and config['observation_model']:
            self.observation_model = config['observation_model']
            self.observation_vector = config['observation_vector']
        else:
            self.observation_model = [(lambda t,p,y: y) for _ in config['observation_vector']]
            self.observation_vector = [[v] for v in config['observation_vector']]
        self.weightings = np.array(config['weightings'][0])
        self.densities = np.array(config['weightings'][1])
        self.regularisation_vector = np.array(config['regularisation_value'])

        self.observation_times = np.array(dataset['t'])
        self.observations = self.observations_from_pandas(dataset['y'])
        self.collocation_matrices = self.colloc_matrices(dataset, model)
        
        self.input_list = [*model.cs, *model.ps]
        self.objective_input_list = [*self.input_list, self.rho, self.alpha]

        self.create_objective(model)
        self.create_objective_functions()

    def create_objective(self, model):
        """ Creates the CasADi objects that represent the objective function
        """
        self.obj_1 = sum(w/len(ov) * ca.sumsqr(self.densities*( 
                            ov - cm@ca.interp1d(model.observation_times,
                            om(model.observation_times, model.ps, 
                               *(model.xs[j] for j in oj)),
                            self.observation_times)
                        ))
                         for om, oj, ov, w, cm in zip(self.observation_model,
                                                      self.observation_vector,
                                                      self.observations,
                                                      self.weightings,
                                                      self.collocation_matrices))
        self.obj_2 = sum(ca.sumsqr((model.xdash[:, i] -
                                    model.model(model.observation_times, *model.cs, *model.ps)[:, i]))
                          for i in range(model.s))/model.n

        self.regularisation = ca.sumsqr((ca.vcat(model.ps) - ca.vcat(self.regularisation_vector))/(1 + ca.vcat(self.regularisation_vector)))

        self.objective = self.obj_1 + self.rho*self.obj_2 + self.alpha*self.regularisation

    def create_objective_functions(self):
        """ Creates functions that expose interior objects for future analysis
        """
        self.obj_fn_1 = ca.Function('fn1', self.input_list, [self.obj_1])
        self.obj_fn_2 = ca.Function('fn2', self.input_list, [self.obj_2])
        self.reg_fn = ca.Function('fn3', self.input_list, [self.regularisation])
        self.obj_fn = ca.Function('objective', self.objective_input_list, [self.objective])

    def observations_from_pandas(self, observations, convert=True):
        """Transposes pandas array to numpy array
        
        Side effect: Converts NaN entries to zeros, if convert is True"""
        arr = np.stack(np.array(observations)).T
        for arr_row in arr:
            if len(arr_row) < self.m:
                arr = np.pad(arr, ((0, 0), (0, self.m-len(arr[0]))), 'constant', constant_values=0)
        if convert:
            arr = np.nan_to_num(arr, copy=True)
        return arr

    def colloc_matrices(self, dataset, model):
        """ Generate the matrix that selects time points that are not NaN in the data
        """

        is_not_nan = lambda x: not np.isnan(x)
        not_nan = [list(map(float, map(is_not_nan, yi))) for yi in dataset['y'].T]
        colloc_matrix_numerical = list(map(np.diag, not_nan))

        return colloc_matrix_numerical
