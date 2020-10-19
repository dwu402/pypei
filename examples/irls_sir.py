"""
SIR fitting via IRLS
"""

import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import colors as colors
from scipy.integrate import solve_ivp
import pypei

###################
# FLAGS
###################
show_truth = False
show_mle = True
do_profile = False
show_profile = True
do_state_uq = False
show_state_uq = True
###################

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

n_obsv = 13
ts = np.linspace(t_span[0], t_span[1]*0.5, n_obsv)
ys = sol_true.sol(ts)[[0,2],:]
y_noisy = ys + 25*np.random.randn(*ys.shape)

if show_truth:
    plt.figure()
    plt.plot(ts, ys.T, '-')
    plt.plot(ts, y_noisy.T, '-o')

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

interpolator = model.all_x_at(model.cs, ts)[:,[0,2]].reshape((-1,1))

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
        },
    ],
    'L': [
        objective._autoconfig_L(y_noisy,),# auto=True, sigma=ss[0]),
        objective._autoconfig_L(model.xs,),# auto=True, sigma=ss[1]),
    ]
}
objective.make(objective_config)

solver = pypei.irls_fitter.Solver(objective=objective)
solver_config = solver.make_config(model, objective)
# solver_config['o'] = pypei.fitter.ipopt_silent
solver.make(solver_config)

x0 = solver.proto_x0(model)
x0x0 = x0['x0']
x0x0[:-2] = x0x0[:-2] * 1000

def p(w):
    return [*w, *[i for l in y_noisy for i in l], 0]

weight_args = {'n_obsv': [n_obsv, model_config['grid_size']]}

sol, ws, shist, whist = solver.irls(x0x0, p, nit=4, weight_args=weight_args, hist=True, MODEL=model)

print("variances: ", pypei.irls_fitter._inverse_weight_functions['gaussian'](ws))
print("pars: ", solver.get_parameters(sol, model))

if show_mle:
    plt.figure()
    plt.plot(model.observation_times, solver.get_state(sol, model))
    plt.plot(ts, y_noisy.T, 'o')

if do_profile:
    prof_config = solver._profiler_configs(model)
    solver.make_profilers(prof_config)
    profiles = solver.profile(sol, p=p, w0=ws, nit=4, weight_args=weight_args)

    if show_profile:
        # ob1kenob = ca.Function('ob1kenob', [solver.decision_vars, solver.parameters], [objective.obj_fn(0)])
        full_ll = ca.Function('lLfn', [solver.decision_vars, solver.parameters], [objective.log_likelihood])
        for i, profile in enumerate(profiles):
            plt.figure()
            plt.plot(profile['ps'], [x['f'] for x in profile['pf']['s']])
            plt.title(f"Weighted obj fn values {i}")
            plt.figure()
            plt.semilogy(profile['ps'], profile['pf']['w'])
            plt.title(f"Weights {i}")
            plt.figure()
            cNorm = colors.Normalize(vmin=profile['ps'][0], vmax=profile['ps'][-1])
            cm = plt.get_cmap('viridis')
            scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

            for n, s in zip(profile['ps'], profile['pf']['s']):
                cl = scalarMap.to_rgba(n)
                x = solver.get_state(s, model)
                S = x[:,0]; I = x[:,1]; R = x[:,2]
                plt.plot(model.observation_times, S, '-', color=cl, alpha=0.7)
                plt.plot(model.observation_times, I, '--', color=cl, alpha=0.7)
                plt.plot(model.observation_times, R, '-.', color=cl, alpha=0.7)
            plt.title(f'State plots {i}')

            plt.figure()
            plt.plot(profile['ps'], [full_ll(s['x'], p(ws)) for s in profile['pf']['s']])
            plt.title(f"Full log likelihood")

if do_state_uq:
    pypei.fitter.reconfig_rto(model, objective, solver, objective_config, index=1)
    rss, rys = solver.gaussian_resample(sol, [y_noisy.flatten(), 0], ws, objective, 1000,
                               weight_args=weight_args, nit=4)

    if show_state_uq:
        plt.figure()
        for y, _ in zip(*rys):
            plt.plot(ts, y.reshape(y_noisy.shape).T, 'k+', alpha=0.1)
        for s, _ in rss:
            plt.plot(model.observation_times, solver.get_state(s, model), 'k', alpha=0.05)
        truth_ts = np.linspace(*t_span, 2001)
        plt.plot(truth_ts, sol_true.sol(truth_ts).T, label="Truth")
        plt.plot(ts, y_noisy.T, 'x', label='Data')
        plt.legend()

        rss_ps = [solver.get_parameters(x, model) for x,_ in rss]
        plt.figure()
        plt.plot(*list(zip(*rss_ps)), 'o')
        plt.plot(*p_true, 'P')
        plt.plot(*solver.get_parameters(sol, model), 'X')

plt.show()
