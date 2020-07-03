import pypei

import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

def turnover_model(t, x, p):
    return [p[0] - p[1]*x[0]]

# using y = x to prevent non-identifiability problems

""" Simulate Ground Truth """
from scipy.integrate import solve_ivp
p_true = [1, 0.1]
y_init = [1]
t_span = [0, 30]

sol_true = solve_ivp(turnover_model, t_span, y_init, args=[p_true], dense_output=True)

# plt.plot(sol_true.t, sol_true.y.T)
# plt.show()

data_t = np.linspace(*t_span, 30)
data_y_true = sol_true.sol(data_t)
data_pd = (data_y_true + np.random.randn(*data_y_true.shape)).reshape(-1,1)

model_form = {
    'state': 1,
    'parameters': 2,
}
model_config = {
    'grid_size': 200,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': t_span,
    'model': turnover_model,    
}

model = pypei.modeller.Model(model_config)

# setting up the objective function
objective = pypei.objective.Objective()

# observation model, now with added interpolation
data_obsv_fn = model.x_at(ca.hcat(model.cs), data_t)

# standard deviation detection
stdev_obsv = ca.SX.sym('so')
stdev_model = ca.SX.sym('sm')

objective_config = {
    'Y': [
        {
            'sz': data_pd.shape,
            'obs_fn': data_obsv_fn,
        },
        {
            'sz': model.xs.shape,
            'unitary': True,
            'obs_fn': objective._MODELFIT(model),
        },
    ],
    'L': [
        {
            'depx': True,
            'x': 1 / stdev_obsv * ca.SX.eye(np.prod(data_pd.shape)),
            'iden': True, 
            'balance': False,
        },
        {
            'depx': True,
            'x': np.sqrt(t_span[-1]/model_config['grid_size']) / stdev_model * ca.SX.eye(np.prod(model.xs.shape)),
            'iden': True, 
            'balance': False,
        }
    ]
}

objective.make(objective_config)

# creating the solver object
solver = pypei.fitter.Solver()
# using default solver setup
solver_config = solver.make_config(model, objective)
solver_config['x'] = ca.vcat([solver_config['x'], stdev_obsv, stdev_model])
solver.make(solver_config)

# initial interate
proto_x0 = solver.proto_x0(model)
# for all ones
# x0 = proto_x0['x0'] 
L0 = np.array([0.1, 0.1]).reshape((-1, 1))
x0 = np.concatenate([proto_x0['c0'], (proto_x0['p0'].T*[1, 1]).T, L0])

# parameters (L matrices and data)
solver.prep_p_former(objective)
y0s = [data_pd, 0]
p = solver.form_p([], y0s)

# bounds on decision variables
# non-negative model parameters
# positive std deviations
lbx = np.concatenate([proto_x0['c0']*-np.inf, [[0], [0], [1e-12], [1e-3]]])
ubx = np.ones(x0.shape)*np.inf
# For some reason, an upper bound needs to be applied. Even so, there are NaN errors, restoration errors, and search direction errors.
# ubx[-2:, 0] = [1e2, 1e2]

# solve
mle_estimate = solver.solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=0)

print(solver.get_parameters(mle_estimate, model))
print(p_true)
print(mle_estimate['x'][-2:])
plt.figure()
plt.plot(model.observation_times, solver.get_state(mle_estimate, model))
plt.plot(data_t, data_pd, 'v')
plt.plot(sol_true.t, sol_true.y.T, 'o')
plt.show()