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

solver = pypei.irls_fitter.Solver()
solver_config = solver.make_config(model, objective)
# solver_config['x'] = ca.vcat([solver_config['x'], ss[0]])
lbx = np.ones(solver_config['x'].shape) * -np.inf
# lbx[-2:,0] = [1e-14, 1e-6]
# lbx[-1] = 1e-14
# ubx = np.ones(solver_config['x'].shape) * np.inf
# ubx[-1:,0] = [1e-6]
solver.make(solver_config)
solver.objective_obj = objective

c_1 = np.ones(model_config['basis_number'])
c_true = np.linalg.pinv(model.phi) @ x
x0_best = [*c_1, *[1, 1, 1]]

def p(w):
    return [*w, *y_noisy, 0]

sol, ws, shist, whist = solver.irls(x0_best, p, nit=10, n_obsv=30, hist=True)

# print(sol, ws)
# print(shist, whist)

print("pars_hist:", [solver.get_parameters(s, model) for s in shist])
print("est. pars:", solver.get_parameters(sol, model))
print("final obsv var:", 1/ws**2)

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

fhist = plt.figure()
for h in np.vstack(whist).T:
    plt.semilogy(h)


plt.show()
