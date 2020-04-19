import pypei

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import casadi as ca

# creation of synthetic underlying truth
p_true = [0.75/10000, 0.075]
y0_true = [10000, 1]
tspan = [0, 50]

def sir_model(t, y, p):
    return [
        -p[0]*y[0]*y[1],
        p[0]*y[0]*y[1] - p[1]*y[1]
    ]

sol_true = solve_ivp(lambda t, y: sir_model(t, y, p=p_true), tspan, y0_true, dense_output=True)

# observation model
# we only see the cumulative cases reported = S(0)-S = integral of bSI
def observation_function(y):
    return y[0,0] - y[:,0]

# construting the data
data_t = np.linspace(0, 30, 11)
data_y = observation_function(sol_true.sol(data_t).T)
data = data_y + np.random.randn(*data_y.shape)*100
# non-negative
data[data < 0] = 0
# strictly increasing
for i,d in enumerate(data):
    if i==0:
        continue
    if d < data[i-1]:
        data[i] = data[i-1]
data_pd = data.reshape(-1,1)

# setting up the basis function model
model_form = {
    'state': 2,
    'parameters': 2,
}
model_config = {
    'grid_size': 200,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': [0, 50],
    'model': sir_model,    
}

model = pypei.modeller.Model(model_config)

# setting up the objective function
objective = pypei.objective.Objective()

# observation model, now with added interpolation
def data_obsv_fn(y):
    return y[0,0] - ca.interp1d(model.observation_times, y[:,0], data_t)

objective_config = {
    'Y': [
        {
            'sz': data_pd.shape,
            'obs_fn':objective._DATAFIT(model, data_obsv_fn),
        },
        {
            'sz': (np.prod(model.xs.shape),1),
            'obs_fn': objective._MODELFIT(model).reshape((-1,1)),
        }
    ],
    'L': [
        objective._autoconfig_L(data_pd),
        objective._autoconfig_L(model.xs),
    ]
}
objective.make(objective_config)

# creating the solver object
solver = pypei.fitter.Solver()
# using default solver setup
solver_config = solver.make_config(model, objective)
solver.make(solver_config)

# initial interate
proto_x0 = solver.proto_x0(model)
# for all ones
# x0 = proto_x0['x0'] 
x0 = np.concatenate([proto_x0['c0'], (proto_x0['p0'].T*[1/10000, 1]).T])

# parameters (L matrices and data)
solver.prep_p_former(objective)
# equivalent to lambda = 2
p = solver.form_p([1/2., 1/1.], [data_pd.T.flatten(), np.zeros((400,1))])

# bounds on decision variables
# non-negative model parameters
lbx = np.concatenate([proto_x0['c0']*-np.inf, [[0], [0]]])

# solve
mle_estimate = solver.solver(x0=x0, p=p, lbx=lbx, lbg=0)