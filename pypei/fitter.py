from . import util
import casadi as ca
import numpy as np

class Solver():
    def __init__(self, config=None):
        self.solver = None
        self.objective_function = None
        self.constraints = None
        self.decision_vars = None
        self.parameters = None

        self.__default_solve_opts__ =  {
            'ipopt': {
                'print_level': 5,
                'print_frequency_iter': 50,
            }
        }
        self.solve_opts = self.__default_solve_opts__

        if config:
            self.make(config)

    def make(self, config):
        self.decision_vars = config['x']
        self.parameters = config['p']
        self.objective_function = config['f']
        self.constraints = config['g']

        if 'o' in config:
            self.solve_opts = config['o']

        self.solver = ca.nlpsol('solver', 'ipopt',
            {
                'x': self.decision_vars,
                'f': self.objective_function,
                'g': self.constraints,
                'p': self.parameters,
            },
            self.solve_opts
        )

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, *args, **kwargs):
        return self.solver.solve(*args, **kwargs)

    @staticmethod
    def make_config(model, objective):
        return {
            'x': ca.vcat([*model.cs, *model.ps]),
            'f': objective.objective_function,
            'g': model.xs.reshape((-1,1)),
            'p': ca.vcat(util.flat_squash(*objective.Ls,*objective.y0s))
        }

    @staticmethod
    def form_p(Ls, y0s):
        pass

    @staticmethod
    def proto_x0(model):
        return {
            'x0': np.ones(ca.vcat([*model.cs, *model.ps]).shape),
            'c0': np.ones(ca.vcat(model.cs).shape),
            'p0': np.ones(ca.vcat(model.ps).shape)
        }
