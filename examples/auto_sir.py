"""
In this example, we solve the classic generalised profiling problem for inference of a deterministic SIR model

In contrast to basic_sir, we also automatically optimise for the covariance matrix
"""

import pypei

import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import casadi as ca

from matplotlib import pyplot as plt

# creation of synthetic, underlying ground truth
p_true = [0.6/10000, 0.25]
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
data_t = np.linspace(0, 28, 15)
data_y = observation_function(sol_true.sol(data_t).T)
data = data_y + np.random.randn(*data_y.shape)*100
# non-negative
# data[data < 0] = 0
# strictly increasing
# data = np.maximum.accumulate(data)
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
def data_obsv_fn(y, p):
    return y[0,0] - ca.interp1d(model.observation_times, y[:,0], data_t)

# standard deviation detection
stdev_obsv = ca.SX.sym('so')
stdev_model = ca.SX.sym('sm')

objective_config = {
    'Y': [
        {
            'sz': data_pd.shape,
            'obs_fn':objective._DATAFIT(model, data_obsv_fn),
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
            'x': 1 / stdev_model * ca.SX.eye(np.prod(model.xs.shape)),
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
L0 = np.array([1/2, 1,]).reshape((-1, 1))
x0 = np.concatenate([proto_x0['c0'], (proto_x0['p0'].T*[1/10000, 1]).T, L0])

# parameters (L matrices and data)
solver.prep_p_former(objective)
y0s = [data_pd, 0]
p = solver.form_p([], y0s)

# bounds on decision variables
# non-negative model parameters
# positive std deviations
lbx = np.concatenate([proto_x0['c0']*-np.inf, [[0], [0], [1e-10], [1e-3]]])
ubx = np.ones(x0.shape)*np.inf

# specify ics
lbx[0] = y0_true[0]
ubx[0] = y0_true[0]


# solve
mle_estimate = solver.solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=0)

# visualise mle solution
if True:
    print(solver.get_parameters(mle_estimate, model))
    print(p_true)
    print(mle_estimate['x'][-2:])
    plt.figure()
    plt.plot(model.observation_times, solver.get_state(mle_estimate, model))
    plt.plot(model.observation_times, observation_function(solver.get_state(mle_estimate, model)))
    plt.plot(data_t, data_obsv_fn(solver.get_state(mle_estimate, model), solver.get_parameters(mle_estimate, model)))
    plt.plot(data_t, data_pd, 'v')
    plt.plot(data_t, data_y, 'x')
    plt.plot(sol_true.t, sol_true.y.T, 'o')
    plt.plot(sol_true.t, observation_function(sol_true.y.T), 'o')
    plt.ylim([0, 15000])
    # plt.show()

plt.show()