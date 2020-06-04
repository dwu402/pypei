"""
In this script, we check that pypei's basic least squares algorithm
is implemented correctly, by checking it against an additive noise model

We take

y ~ N(0, sigma^2)

"""
import pypei

import numpy as np
from matplotlib import pyplot as plt

# generate sigma = 0.3
y = np.random.randn(50) * 0.3

def zero_model(t, y, p):
    return [0.*y[0]]

model_form = {
    'state': 1,
    'parameters': 0,
}
model_config = {
    'grid_size': 50,
    'basis_number': 20,
    'model_form': model_form,
    'time_span': [0, 50],
    'model': zero_model,    
}

model = pypei.modeller.Model(model_config)

objective = pypei.objective.Objective()

objective_config = {
    'Y': [
        {
            'sz': y.shape,
            'obs_fn':objective._DATAFIT(model, lambda x: 0.*x),
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
sds = np.logspace(-1, 1, num=100)
estimates = []
for std_dev in sds:
    p = solver.form_p([1/std_dev], [y])
    estimates.append(solver.solver(x0=x0, p=p))

plt.semilogx(sds, [e['f'] for e in estimates])
print(np.std(y))
i = np.argmin([e['f'] for e in estimates])
print(sds[i])
print(np.sqrt(np.linalg.norm(y)**2/len(y)))
plt.show()
