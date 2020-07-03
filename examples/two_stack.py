""" In this script we solve a stacked nonlinear regression problem

y(t) = g(x(t), p) + e_y
x(t) = f(t, p) + e_x

e_y ~ N(0, s_y^2 * I)
e_x ~ N(0, s_x^2 * I)

We note that this can be combined as

y(t) = g(f(t, p) + e_x, p) + e_y

which in the case where g is linear can be formulated as

y(t) = h(t, p) + e_y*

However, we will tackle this problem by stacking the model:

y   =     g(x, p)     +   e_y
0   =   x - f(t, p)   +   e_x

For this example, we will use the forms

y = a*x + e_y
x = b*t**2 + c*t + e_x

where we wish to estimate a, b, c, and variances of e_y, e_x

Here, we will use modeller to represent x, and also estimate it as
a nuisance parameter.
"""
import numpy as np
from matplotlib import pyplot as plt
import casadi as ca
import pypei

visual = True

t_span = [0, 10]
n_obsv = 30
t = np.linspace(*t_span, n_obsv)

def f(t, p):
    return p[0]*t**2 + p[1]*t

def g(x, p):
    return x - p[2]

p_true = [0.2, 1.1, 3.3]

x = f(t, p_true)
y = g(x, p_true)

x_noisy = x + np.random.randn(*t.shape)
y_noisy = g(x_noisy, p_true) + np.random.randn(*t.shape)

# Model representation of x
model_form = {
    'state': 1,
    'parameters': 3,
}
model_config = {
    'grid_size': n_obsv,
    'basis_number': 30,
    'model_form': model_form,
    'time_span': t_span,
    'model': lambda t, x, p: [f(t, p)],
}

model = pypei.modeller.Model(model_config)

# Construct Objective
objective = pypei.objective.Objective()

# symbolic representations of parameters and std devs
ps = model.ps
ss = ca.SX.sym('s', 2)

objective_config = {
    'Y': [
        {
            'sz': y_noisy.shape,
            'obs_fn': g(model.xs, ps).reshape((-1, 1)),
        },
        {
            'sz': model.xs.shape,
            'unitary': True,
            'obs_fn': (model.xs - model.model(model.observation_times,
                                             *model.cs, *model.ps)).reshape((-1, 1)),
        }
    ],
    'L': [
        objective._autoconfig_L(y_noisy, auto=True, sigma=ss[0]),
        objective._autoconfig_L(x_noisy, auto=True, sigma=ss[1]),
    ]
}

objective.make(objective_config)

solver = pypei.fitter.Solver()
solver_config = solver.make_config(model, objective)
solver_config['x'] = ca.vcat([solver_config['x'], ss])
lbx = np.ones(solver_config['x'].shape) * -np.inf
lbx[-2:,0] = [1e-14, 1e-14]
solver.make(solver_config)

sol = solver(x0=np.ones(solver_config['x'].shape, dtype=float)*0.5, p=[*y_noisy, 0], lbx=lbx)

print(sol['f'])
print(sol['x'])

if visual:
    xe = solver.get_state(sol, model)
    pe = solver.get_state(sol, model)
    ye = g(xe, pe)

    fx = plt.figure()
    plt.plot(t, x, label="True", )
    plt.plot(t, x_noisy, 'o', label="Noisy Data", )
    plt.plot(t, xe, label='Estimated', )
    plt.title("Underlying State", )
    plt.legend()

    fy = plt.figure()
    plt.plot(t, y, label='True', )
    plt.plot(t, y_noisy, 'o', label='Noisy Data', )
    plt.plot(t, ye, label='Estimated', )
    plt.title("Observables", )
    plt.legend()

    plt.show()
    