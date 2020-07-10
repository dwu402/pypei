"""
In this script we use just the objective and fitter modules to 
construct a simple nonlinear regression problem.

The idea is to automate the estimation of both parameters and
noise.

We will be fitting synthetic data. The functional form of 
the regression is:

y = a * x^2 + b + e
e ~ N(0, s^2 * I)
"""

import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
import pypei

visual = True

t_span = [0, 10]
n_obsv = 30
x = np.linspace(*t_span, n_obsv)

def functional(x, p):
    return p[0]*x**2 + p[1]

p_true = [0.2, 4]
y = np.array(functional(x, p_true)).T
# add noise ~ N(0, 1)
y_noisy = y + np.random.randn(*y.shape)

if visual:
    print(y_noisy.shape)
    plt.plot(x, y, '-', label='True')
    plt.plot(x, y_noisy, 'o', label='Observed')

# symbolic representing the regressors
ps = ca.SX.sym('p', 2)
# symbolic representing the standard deviation
# here, we represent the std devs over the entire time window
# into separate chunks to estimate (though we know them to be the same)
std_dev_primitives = ca.SX.sym('s', 3) # 3 chunks
std_dev = ca.repmat(std_dev_primitives, 1, 10).T.reshape((-1,1))
# print(std_dev)

objective = pypei.objective.Objective()
objective_config = {
    'Y': [
        {
            'sz': y_noisy.shape,
            'obs_fn': functional(x, ps),
        },
    ],
    'L': [
        objective._autoconfig_L(y_noisy, auto=True, sigma=std_dev)
    ]
}

objective.make(objective_config)

solver = pypei.fitter.Solver()
solver_config =  {
    # decision variables
    'x': ca.vcat([ps, std_dev_primitives]),
    # functional form of the -ve log likelihood
    'f': objective.objective_function,
    # constraints
    'g': [],
    # non-decision variables of the log likelihood (data)
    'p': ca.vcat(pypei.functions.misc.flat_squash(*objective.Ls, *objective.y0s))
}
solver.make(solver_config)

# initialising parameters and noise as 0.3, giving noisy data.
sol = solver(x0=np.ones(solver_config['x'].shape, dtype=float)*0.3, p=y_noisy)

print(sol['x'])
if visual:
    plt.plot(x, functional(x, sol['x']), label='Estimated')
    plt.legend()
    plt.show()