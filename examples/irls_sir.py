"""
SIR fitting via IRLS
"""

import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import pypei

def sir_model(t, y, p):
    b, a = p[:2]
    s, i, r = y[:3]
    return [
        -b*s*i/(s+i+r),
        b*s*i/(s+i+r) - a*i,
        a*i,
    ]

t_span = [0, 20]
y0_true = [999, 1, 0]
p_true = [1.3, 0.2]

sol_true = solve_ivp(sir_model, t_span=t_span, y0=y0_true, args=[p_true], dense_output=True)

n_obsv = 21
ts = np.linspace(*t_span, n_obsv)
ys = sol_true.sol(ts)
y_noisy = ys + 20*np.random.randn(*ys.shape)

# plt.plot(ts, ys.T, '-')
# plt.plot(ts, y_noisy.T, '-o')
# plt.show()

model_form = {
    'state': 3,
    'parameters': 2,
}
model_config = {
    'grid_size': 200,
    'basis_number': 42,
    'model_form': model_form,
    'time_span': t_span,
    'model': sir_model,
}
model = pypei.modeller.Model(model_config)

interpolator = model.all_x_at(model.cs, ts).reshape((-1,1))

objective = pypei.objective.Objective()
objective_config = {
    'Y': [
        {
            'sz': y_noisy.shape,
            'obs_fn': interpolator,
        },
        {
            'sz': model.xs.shape,
            'unitary': True,
            'obs_fn': objective._MODELFIT(model),
        }
    ],
    'L': [
        objective._autoconfig_L(y_noisy,),# auto=True, sigma=ss[0]),
        objective._autoconfig_L(model.xs,),# auto=True, sigma=ss[1]),
    ]
}
objective.make(objective_config)

solver = pypei.irls_fitter.Solver()
solver_config = solver.make_config(model, objective)
solver.make(solver_config)
solver.objective_obj = objective

# lbx = np.ones(solver_config['x'].shape) * -np.inf

x0 = solver.proto_x0(model)
x0x0 = x0['x0']
x0x0[:-2] = x0x0[:-2] * 1000

def p(w):
    return [*w, *[i for l in y_noisy for i in l], 0]

sol, ws, shist, whist = solver.irls(x0x0, p, nit=10, n_obsv=n_obsv, hist=True)

print("variances: ", pypei.irls_fitter._inverse_weight_functions['gaussian'](ws))
print("pars: ", solver.get_parameters(sol, model))

plt.plot(model.observation_times, solver.get_state(sol, model))
plt.plot(ts, y_noisy.T, 'o')
plt.show()