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

y = x + a + e_y
x = b*t**2 + c*t + e_x

where we wish to estimate a, b, c, and variances of e_y, e_x

Here, we will use modeller to represent x, and also estimate it as
a nuisance parameter.

One problem with this setup is that the variances in the errors are
nonidentifiable, since g is linear, meaning that we can represent 
the problem as

y = b*t**2 + c*t + a + e_yx, e_yx ~ N(0, (s_y^2 + s_x^2)*I)
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
    return p[0]*t**2 + p[1]*t

def g(x, p):
    return x**2 + p[2]

p_true = [0.2, 1.1, 3.3]

x = f(t, p_true)
y = g(x, p_true)

x_noisy = x + np.random.randn(*t.shape)
y_noisy = g(x_noisy, p_true) + 2*np.random.randn(*t.shape)

# Model representation of x
model_form = {
    'state': 1,
    'parameters': 3,
}
model_config = {
    'grid_size': n_obsv,
    'basis_number': 20,
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
            'obs_fn': (model.xs - f(model.observation_times, model.ps)).reshape((-1, 1)),
        }
    ],
    'L': [
        objective._autoconfig_L(y_noisy,),# auto=True, sigma=ss[0]),
        objective._autoconfig_L(y_noisy,),# auto=True, sigma=ss[1]),
    ]
}

objective.make(objective_config)

solver = pypei.fitter.Solver()
solver_config = solver.make_config(model, objective)
# solver_config['x'] = ca.vcat([solver_config['x'], ss[0]])
lbx = np.ones(solver_config['x'].shape) * -np.inf
# lbx[-2:,0] = [1e-14, 1e-6]
# lbx[-1] = 1e-14
# ubx = np.ones(solver_config['x'].shape) * np.inf
# ubx[-1:,0] = [1e-6]
solver.make(solver_config)

c_1 = np.ones(model_config['basis_number'])
c_true = np.linalg.pinv(model.phi) @ x
x0_best = [*c_1, *[1, 1, 1]]
x0 = x0_best
ss1 = 1
ss2 = 1
converged = False
f1 = ca.Function('L_O_L', [solver_config['x'], solver_config['p']], [objective.us_obj_fn(0)])
f2 = ca.Function('L_O_L', [solver_config['x'], solver_config['p']], [objective.us_obj_fn(1)])
s1hist, s2hist = [ss1], [ss2]
errhist = []
for _ in range(10):
    pvec = [ss1, ss2, *y_noisy, 0]
    # print(ss1, ss2)
    sol = solver(x0=x0, p=pvec, lbx=lbx)
    x0 = sol['x'].toarray().flatten()

    stemp1 = 1/ca.sqrt(f1(sol['x'], pvec)/30)
    stemp2 = 1/ca.sqrt(f2(sol['x'], pvec)/30)

    err = np.abs(np.linalg.norm([float(stemp1), float(stemp2)]) - np.linalg.norm([ss1, ss2])) / (np.abs(np.linalg.norm([ss1, ss2])) + 1)
    errhist.append(err)

    print("err:", err)
    if err < 1e-6 or (len(errhist) > 1 and np.abs(err-errhist[-2])/(np.abs(errhist[-2])+1) < 1e-5):
        converged = True

    else:
        # Crank-Nicholson iteration
        omega = 1
        ss1 = float((1-omega)*ss1 + omega*stemp1)
        ss2 = float((1-omega)*ss2 + omega*stemp2)

        s1hist.append(ss1)
        s2hist.append(ss2)

    print("obj val:", sol['f'])
    # print(sol['x'])
    print("est. pars:", solver.get_parameters(sol, model))
    # print("est stddevs:", sol['x'][-1:], ", sum vars:", sol['x'][-1]**2 + ss1)
    print("obsv var:", ca.sumsqr(y_noisy-y)/30)
    print("est var: ", 1/ss1**2, 1/ss2**2)

print("final obsv var:", 1/stemp1**2, 1/stemp2**2)

if visual:
    xe = solver.get_state(sol, model)
    pe = solver.get_parameters(sol, model)
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

    fss = plt.figure()
    plt.semilogy(s1hist, label='Obsv Weight')
    plt.semilogy(s2hist, label='Model Weight')
    plt.legend()

    plt.show()
