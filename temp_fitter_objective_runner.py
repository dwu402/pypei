import modeller
import fitter

import casadi as ca
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt


p_true = [0.0005, 0.1]
tspan = [0, 20]
def system(t, y, p):
    return [
        -p[0]*y[0]*y[1],
        p[0]*y[0]*y[1] - p[1]*y[1],
        p[1]*y[1]
    ]
sol = solve_ivp(lambda t, y: system(t, y, p_true), tspan, [9999, 1, 0])

data = sol.y[-1, :]
data_pd = np.array([[i] for i in data])
config = {
    "grid_size": 49,
    "basis_number": 13,
    "model_form": {
        "state": 3,
        "parameters": 2
    },
    "time_span": tspan,
    "knot_function": None,
    "model": system,
    "dataset": {"y": data_pd, "t": sol.t},
    "observation_vector": [2],
    "weightings":[
        [1]*3,
        [1]*len(sol.t)
    ],
    "regularisation_vector" : p_true,
}

model = modeller.Model(config)
objective = fitter.Objective()
objective.make(config, config['dataset'], model)

solver_setup = {
    'f': objective.objective,
    'x': ca.vcat(objective.input_list),
    'p': ca.hcat([objective.rho, objective.alpha])
}

solver = ca.nlpsol('solver', 'ipopt', solver_setup)

p0 = np.ones(2)
c0 = [np.ones(13) for _ in range(3)]
x0 = np.hstack([p0, *c0])

prange = np.logspace(-6, 6, 50)
solutions = []
xguess = x0
for p in prange:
    solutions.append(solver(x0=xguess, p=[p, 1e-4], lbx=0))
    xguess = np.array(solutions[-1]['x']).flatten()

# recover_traj = ca.Function('recx', [solver_setup['x']], model.get_x_obsv())

# plt.plot(model.observation_times, recover_traj(solution['x'])[2])
# plt.plot(sol.t, data, 'o')
# plt.show()