"""
In this script, we check that pypei's basic least squares algorithm
is implemented correctly, by checking it against an additive noise model

We take

y ~ N(0, sigma^2)

"""
import pypei

import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

# generate sigma = 0.3
y = np.random.randn(50) * 0.3

def zero_model(t, y, p):
    return [0.*y[0]]

model_form = {
    'state': 1,
    'parameters': 0,
}
model_config = {
    'grid_size': 50,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': [0, 50],
    'model': zero_model,    
}

model = pypei.modeller.Model(model_config)

objective = pypei.objective.Objective()

stdev_sym = ca.SX.sym('s')

objective_config = {
    'Y': [
        {
            'sz': y.shape,
            'obs_fn':objective._DATAFIT(model, lambda x: 0.*x),
        },
    ],
    'L': [
        {
            'depx': True,
            'x': 1 / stdev_sym * ca.SX.eye(np.prod(y.shape)),
            'iden': True, 
            'balance': False,
        }
    ]
}
objective.make(objective_config)

solver = pypei.fitter.Solver()
# using default solver setup
solver_config = solver.make_config(model, objective)
solver_config['x'] = ca.vcat([solver_config['x'], stdev_sym])
solver.make(solver_config)

# initial interate
x0 = np.ones(solver_config['x'].shape)

# parameters (L matrices and data)
solver.prep_p_former(objective)
p = solver.form_p([], [y])
sol = solver.solver(x0=x0, p=p)

print(np.std(y))
print(np.sqrt(np.linalg.norm(y)**2/len(y)))
print(sol['x'][-1])
