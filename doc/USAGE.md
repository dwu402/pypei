# Usage

## Overview

This package is used for the inference of parameters and states of oridinary differential equation models from data.

The general construction of the problem is as follows:

- Specify the differential equation model
- Project the model onto a B-spline basis
- Construct the objective (log-likelihood) referencing the projected model
- Construct the nonlinear objective solver that solves the given objective

Though there are a few helper functions, the package is relatively immature, and decent amount of boilerplate to use.

## Defining the model and state space

We define the model as with any other differential equation model that would go into, for example, the `scipy.integrate.solve_ivp` function. It should be of the following form:

```python
def ode_model(t, y, p):
    """ Some differential equation model

    Arguments
    ---------
    t : float
        Time / dependent variable
    y : iterable
        State vector
    p : iterable
        Parameter vector

    Returns
    -------
    iterable representing dy/dt

    """
    return [dydt(0), dydt(1), ...]
```

We then can pass this model into the B-spline projection module, `pypei.modeller`.

The `model` requires an input configuration, which is documented in the `Model.generate_model` method. We can use the following as a general-use configuration/boilerplate:

```python
model_form = {
    'state': int(),                         # length of state vector
    'parameters': int(),                    # length of parameter vector
}

time_span = [
    float(),                                # start time of integration
    float(),                                # end time of integration
]

model_configuration = {
    'grid_size': int(),                     # size of collocation grid
    'basis_number': int(),                  # number of B splines to project onto
    'model_form': model_form,               # information about the ODE
    'time_span': time_span,                 # bounds on time for integration
    'model': ode_model,                     # the ODE model function from above
}

model = pypei.modeller.Model(model_config)
```

## The objective

We assume an objective function form of:

$$ H(x, y) = \sum_i\left\{|| L_i(y_i -x_i) ||^2_2 - \chi(L_i) \right\}$$

which reduces to the Gaussian log-likelihood when $\chi(L_i) = 2\log |L_i|$, where $L_i^TL_i = \Gamma^{-1}_i$.

We can use the `pypei.Model` object to represent $x_i$, where $x_i$ may be select states of the model, their derivatives, or other functions.

The `pypei.Objective` object, like the `Model` object, also requires a configuration input. This is split in two sections:

- the definition of the $(y_i - x_i)$ components, and
- the definition of the $L_i$ components

These are the `Y` and `L` fields respectively - both fields take list values that contain individual components.

The `Y` field has the structure:

```
Y:
    sz: shape of x_i and y_i
    obs_fn: the object that represents x_i
    unitary: boolean that repsents whether all values of y_i are identical
```

The `L` field has the structure:

```
L:
    TODO
```
There are helper functions that can automate some of these specifications.

TODO

## The solver

`pypei.fitter.Solver` and `pypei.irls_fitter.Solver` provide interfaces to the underlying CasADi interface to various nonlinear programming utilities. By default, this will be IPOPT.

`pypei.fitter` is more of a direct interface that solves the objective as an optimisation problem. This is useful if the $L_i$ components are all known.

`pypei.irls_fitter` implements the iteratively reweighted least squares procedure, which is used in generalised linear model fitting. This is use to help automate the determination of appropriate $L_i$ if they are unknown to the user.

TODO

## Likelihood profiling and Uncertainty quantification

The solver modeuls include helper functions for performing
- likelihood profiling for uncertainty quantification of parameters
- randomise-then-optimise for uncertainty quantification of states and parameters

TODO

## Extracting and using output

The returned objects from the solvers are typically just the dictionary outputs from the CasADi API. The `irls_fitter` can provide histories and other diagnostic information if the `hist` flag is set when running the solver.

The solver provides some utility functions for extracting states and parameters from the output dictionary:

- `Solver.get_state(solution, model)`
- `Solver.get_parameters(solution, model)`

TODO