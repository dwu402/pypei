import numpy as np

def model(t, state, parameters):

    x, y, z = state
    r, a, b, k, p, m, s, c, d, f, g, j, l = parameters

    return [
        r*x - a*x*(b-x) - k*x*y,
        p*x/(m**2+x**2) + s*(y**3)/(c**3+y**3) - d*y,
        f*y*z - g*z - j*x - l*y
    ]

def model_form():
    return {
        "state": 3,
        "parameters": 13,
    }
