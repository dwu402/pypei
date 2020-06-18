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

N = 100
dt = 100/N

# generate sigma = 0.3
y = np.random.randn(N) * 1.5

def zero_model(t, y, p):
    return [0.*y[0]]

model_form = {
    'state': 1,
    'parameters': 0,
}
model_config = {
    'grid_size': N,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': [0, 100],
    'model': zero_model,
}

model = pypei.modeller.Model(model_config)

objective = pypei.objective.Objective()

stdev_sym = ca.SX.sym('s')

# standard deviation detection
stdev_obsv = ca.SX.sym('so')
stdev_model = ca.SX.sym('sm')

objective_config = {
    'Y': [
        {
            'sz': y.shape,
            'obs_fn':objective._DATAFIT(model),
        },
        {
            'sz': model.xs.shape,
            'unitary': True,
            'obs_fn': objective._MODELFIT(model),
        },
    ],
    'L': [
        objective._autoconfig_L(y, auto=True, sigma=stdev_obsv),
        {
            'depx': True,
            'x': 1 / stdev_model * ca.SX.eye(np.prod(model.xs.shape)),
            'diag': True,
            'balance': False,
        }
    ]
}
objective.make(objective_config)

solver = pypei.fitter.Solver()
# using default solver setup
solver_config = solver.make_config(model, objective)
solver_config['x'] = ca.vcat([solver_config['x'], stdev_obsv, stdev_model])
solver.make(solver_config)

# initial interate
x0 = np.ones(solver_config['x'].shape)
x0[:-2] = 0

# lower bounds on variance
lbx = np.ones(x0.shape) * -np.inf
# lbx[-1] = 1e-7

# parameters (L matrices and data)
solver.prep_p_former(objective)
p = solver.form_p([np.sqrt(dt)], [y, 0])
cough = []
coughspace = np.logspace(-10, 0, num=21)
for sm in coughspace:
    lbx[-1] = sm
    sol = solver.solver(x0=x0, p=p, lbx=lbx)
    cough.append(sol['x'][-2:])
# sol = solver(x0=x0, p=p, lbx=lbx)

print(np.std(y))
print(np.sqrt(np.linalg.norm(y)**2/len(y)))
# print(sol['x'][-2:])
print(cough)
plt.loglog(coughspace, [c.toarray().flatten()[0] for c in cough], 'o', color='b')
plt.axhline(np.sqrt(np.linalg.norm(y)**2/len(y)), linestyle='--', color='k')
plt.axhline(np.std(y), color='r')
plt.show()