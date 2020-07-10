"""
two_stack as one
"""

import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
import pypei

np.random.seed(None)

visual = True
# visual = False

t_span = [0, 10]
n_obsv = 30
t = np.linspace(*t_span, n_obsv)

def f(t, p):
    return p[0]*t**2 + p[1]*t + p[2]

p_true = [0.2, 1.1, 3.3]

y = f(t, p_true)

y_noisy = y + np.random.randn(*t.shape)

objective = pypei.objective.Objective()

ps = ca.SX.sym('p', 3)
sigma = ca.SX.sym('s')

objective_config = {
    'Y': [
        {
            'sz': y_noisy.shape,
            'obs_fn': f(t, ps),
        },
    ],
    'L': [
        objective._autoconfig_L(y_noisy, auto=True, sigma=sigma),
    ]
}

objective.make(objective_config)

solver = pypei.fitter.Solver()

solver_config = {
    'x': ca.vcat([ps, sigma]),
    'f': objective.objective_function,
    'g': [],
    'p': ca.vcat(pypei.functions.misc.flat_squash(*objective.Ls, *objective.y0s))
}
solver.make(solver_config)

sol = solver(x0=np.ones(solver_config['x'].shape, dtype=float)*0.3, p=y_noisy)

print(sol['x'])
if visual:
    plt.plot(t, y, label='Truth')
    plt.plot(t, y_noisy, 'o', label='Data')
    plt.plot(t, f(t, sol['x'][:-1]), label='Estimated')
    plt.legend()
    plt.show()