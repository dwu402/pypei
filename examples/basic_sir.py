import pypei

import numpy as np
from scipy.integrate import solve_ivp
import casadi as ca

from matplotlib import pyplot as plt

# Flags for future
known_initial_susceptible_size = True
visualise_mle = True


# creation of synthetic underlying truth
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
            'sz': 0,
            'obs_fn': objective._MODELFIT(model),
        },
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
p = solver.form_p([1/2., 0.], [data_pd.T.flatten(), 0])

# bounds on decision variables
# non-negative model parameters
lbx = np.concatenate([proto_x0['c0']*-np.inf, [[0], [0]]])
ubx = proto_x0['x0']*np.inf

# specify ics if known
if known_initial_susceptible_size:
    lbx[0] = y0_true[0]
    ubx[0] = y0_true[0]

# solve
mle_estimate = solver.solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=0)

# visualise mle solution
if visualise_mle:
    print(solver.get_parameters(mle_estimate, model))
    print(p_true)

    plt.plot(model.observation_times, solver.get_state(mle_estimate, model))
    plt.plot(model.observation_times, observation_function(solver.get_state(mle_estimate, model)))
    plt.plot(data_t, data_obsv_fn(solver.get_state(mle_estimate, model)))
    plt.plot(data_t, data_pd, 'v')
    plt.plot(data_t, data_y, 'x')
    plt.plot(sol_true.t, sol_true.y.T, 'o')
    plt.plot(sol_true.t, observation_function(sol_true.y.T), 'o')
    plt.ylim([0, 15000])
    plt.show()

# profile likelihood for parameter uncertainty
profiler_configs = solver.make_profiler_configs(model)
solver.make_profilers(profiler_configs)

# run profilers
profiles = solver.profile(mle=mle_estimate, p=p, lbx=lbx, ubx=ubx, lbg=0)

# predictive uncertainty
resample_config = dict()
resample_sols = []
resamples = pypei.util.resample_data(data_pd, resample_config, n=50)
for resample in resamples:
    p = solver.form_p([1/2., 1/1.], [resample.T.flatten(), 0])
    # resample_sols.append(solver.solver(x0=x0, p=p, lbx=lbx, lbg=0))
