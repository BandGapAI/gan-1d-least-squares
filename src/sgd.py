from gan import Jgh

import numpy as np


def J_diff_g_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (Jgh(x, z, a, b, c, g + delta, h) - Jgh(x, z, a, b, c, g - delta, h))/(2*delta)


def J_diff_h_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (Jgh(x, z, a, b, c, g, h + delta) - Jgh(x, z, a, b, c, g, h - delta))/(2*delta)


def _get_parameters(config, g2, h2):
    a, b, c, g, h = [v[1] for v in config.items()]
    g = g2 if g2 is not None else g
    h = h2 if h2 is not None else h
    return a, b, c, g, h


def calc_J_diff(config, x, z, method=J_diff_g_numerical, g2=None, h2=None):
    a, b, c, g, h = _get_parameters(config, g2, h2)
    return np.array([[method(x, z, a, b, c, g_, h_) for g_ in g] for h_ in h])


def SGD_step_gh(x, z, a, b, c, g0, h0, eps_g, eps_h):
    g = g0 - eps_g*J_diff_g_numerical(x, z, a, b, c, g0, h0)
    h = h0 - eps_h*J_diff_h_numerical(x, z, a, b, c, g0, h0)
    return g, h


def SGD_gh(x, z, a, b, c, g0, h0, eps_g, eps_h, stop):
    g, h = g0, h0
    for _ in range(stop):
        g, h = SGD_step_gh(x, z, a, b, c, g, h, eps_g, eps_h)
        yield g, h
