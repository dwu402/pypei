"""
In this example, we solve the classic generalised profiling problem for inference of a deterministic SIR model

The objective function can be expressed as

H(c, p | s0, s1, y) = 1/(s0)^2 ||y - g(Phi c)||^2 + 1/(s1)^2 || D(Phi c) - f(Phi c, p) ||

for the SIR model

Dx = f(x, p)

and the observation model

y = g(x) + e, e ~ N(0, s0^2)

where 
D is a differential operator (in this case d/dt)
f is the vector field of the SIR model (ODE RHS)
p are the model parameters

y is the observed data
g is the observation function

c is the projection of the state estimate onto the basis
Phi is the basis of projection

s0 is the estimated standard deviation of the error in the observation model
s1 is the estimated standard deviation of the error in the SIR model

-------------------------------------

We generate synthetic data under the observation model, and partially observe it.

This means that not all states are observed

The observation function is

y(t) = S(0) - S(t)

Further, we assume that we do not data to the end of the epidemic.

--------------------------------------

The objective is to

1. Compute an MLE of the state and parameters, given the partially observed data
2. Quantify the amount of uncertainty in the analysis
2. Compute uncertainty intervals on the parameter estimates
3. Compute uncertainty intervals on the state

"""

import pypei

import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import casadi as ca

from matplotlib import pyplot as plt

# Flags for future
known_initial_susceptible_size = True
visualise_mle = True
profile = True
visualise_profile = True
lcurve = True
visualise_lcurve = True
predictive_uq = True
visualise_predict = True


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
def data_obsv_fn(y):
    return y[0,0] - ca.interp1d(model.observation_times, y[:,0], data_t)

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
y0s = [data_pd, 0]
p = solver.form_p([1/2., 1.], y0s)

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
    plt.figure()
    plt.plot(model.observation_times, solver.get_state(mle_estimate, model))
    plt.plot(model.observation_times, observation_function(solver.get_state(mle_estimate, model)))
    plt.plot(data_t, data_obsv_fn(solver.get_state(mle_estimate, model)))
    plt.plot(data_t, data_pd, 'v')
    plt.plot(data_t, data_y, 'x')
    plt.plot(sol_true.t, sol_true.y.T, 'o')
    plt.plot(sol_true.t, observation_function(sol_true.y.T), 'o')
    plt.ylim([0, 15000])
    # plt.show()

# profile likelihood for parameter uncertainty
if profile:
    profiler_configs = solver._profiler_configs(model)
    solver.make_profilers(profiler_configs)

    # correctly estimate variances empirically
    variances = pypei.fitter.estimate_variances(objective, solver, mle_estimate, y0s)
    p_pr = solver.form_p([1/np.sqrt(float(v)) for v in variances], y0s)

    # run profilers
    profiles = solver.profile(mle=mle_estimate, p=p_pr, lbx=lbx, ubx=ubx, lbg=0)

    if visualise_profile:
        for profile in profiles:
            plt.figure()
            fpeak = min([pf['f'] for pf in profile['pf']])
            plt.plot(profile['ps'], [(pf['f']-fpeak) for pf in profile['pf']])
        # plt.show()

# generating an L curve
if lcurve:
    f1f2 = ca.Function("f1f2", [solver.decision_vars, solver.parameters], [objective.us_obj_fn(0), objective.us_obj_fn(1)])
    # profile over first L
    L1 = np.logspace(-4, 3, num=71)
    L1_profile = []
    lcrv = []
    xi = mle_estimate['x']
    for Li in L1:
        pl = solver.form_p([Li, 1], y0s)
        L1_profile.append(solver.solver(x0=xi, p=pl, lbx=lbx, ubx=ubx, lbg=0))
        xi = L1_profile[-1]['x']
        lcrv.append(f1f2(L1_profile[-1]['x'], pl))
    if visualise_lcurve:
        fpeak = min([s['f'] for s in L1_profile])
        plt.figure()
        plt.loglog(L1, [(s['f']-fpeak) for s in L1_profile])
        # plt.show()

# predictive uncertainty: simple data resampling
if predictive_uq:
    pypei.fitter.reconfig_rto(model, objective, solver, objective_config, index=1)

    resample_sols = []
    resamples = pypei.fitter.gaussian_resampling(objective, solver, mle_estimate, y0s, num=50)
    for resample, gpr in resamples:
        # resample[resample < 0] = 0
        # resample = np.maximum.accumulate(resample)
        p = solver.form_p([1/2., 1/1.], [resample, gpr])
        resample_sols.append(solver.solver(x0=mle_estimate['x'], p=p, lbx=lbx, ubx=ubx, lbg=0))

    if visualise_predict:
        plt.figure()
        plt.violinplot([observation_function(solver.get_state(s, model))[-1] for s in resample_sols])
        plt.figure()
        for s in resample_sols:
            plt.plot(model.observation_times, observation_function(solver.get_state(s, model)))
        plt.plot(sol_true.t, observation_function(sol_true.y.T), 'ko')
        # plt.show()

plt.show()