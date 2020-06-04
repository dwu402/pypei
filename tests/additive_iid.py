"""
In this script, we check that pypei's basic least squares algorithm
is implemented correctly, by checking it against an additive noise model

We take

y ~ N(theta*x+mu, sigma^2)

"""
import pypei

import numpy as np
from matplotlib import pyplot as plt

def linear_model(t, y, p):
    return [p[0]*y[0]]


# generate y = 0.7*x, sigma = 0.3
# 
t_span = (0, 20)
x = np.linspace(*t_span, 100)
y = 0.7 * x + np.random.randn(100) * 0.3


model_form = {
    'state': 1,
    'parameters': 1,
}
model_config = {
    'grid_size': 100,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': t_span,
    'model': linear_model,    
}

model = pypei.modeller.Model(model_config)

objective = pypei.objective.Objective()

data_obsv_fn = lambda x, p: p[0]*x

objective_config = {
    'Y': [
        {
            'sz': y.shape,
            'obs_fn': data_obsv_fn(model.observation_times, model.ps).reshape((-1, 1)),
        },
    ],
    'L': [
        objective._autoconfig_L(y),
    ]
}
objective.make(objective_config)

solver = pypei.fitter.Solver()
# using default solver setup
solver_config = solver.make_config(model, objective)
solver.make(solver_config)

# initial interate
proto_x0 = solver.proto_x0(model) 
x0 = proto_x0['x0']

# parameters (L matrices and data)
solver.prep_p_former(objective)

# profile over different std. deviations
sds = np.logspace(-1, 1, num=30)
estimates = []
for std_dev in sds:
    p = solver.form_p([1/std_dev], [y])
    estimates.append(solver.solver(x0=x0, p=p,))

plt.semilogx(sds, [e['f'] for e in estimates])
opt = np.argmin([e['f'] for e in estimates])
print("Us: s= ", sds[opt], ", th= ", solver.get_parameters(estimates[opt], model))
r = (y-((x@y)/(x@x))*x)
print("MLE: s= ", np.sqrt((r@r)/100), ", th= ", (x@y)/(x@x))
plt.show()