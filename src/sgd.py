from gan import Jgh

import numpy as np


def J_diff_a_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (Jgh(x, z, a + delta, b, c, g, h) - Jgh(x, z, a - delta, b, c, g, h))/(2*delta)


def J_diff_b_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (Jgh(x, z, a, b + delta, c, g, h) - Jgh(x, z, a, b - delta, c, g, h))/(2*delta)


def J_diff_h_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (Jgh(x, z, a, b, c, g, h + delta) - Jgh(x, z, a, b, c, g, h - delta))/(2*delta)


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


def SGD_step(x, z, a0, b0, c, g0, h0, eps=0.1):
    a = a0 + eps*J_diff_a_numerical(x, z, a0, b0, c, g0, h0)
    b = b0 + eps*J_diff_b_numerical(x, z, a0, b0, c, g0, h0)
    g = g0 - eps*J_diff_g_numerical(x, z, a0, b0, c, g0, h0)
    h = h0 - eps*J_diff_h_numerical(x, z, a0, b0, c, g0, h0)
    return a, b, g, h


def SGD(x, z, a0, b0, c, g0, h0, stop):
    a, b, g, h = a0, b0, g0, h0
    for _ in range(stop - 1):
        a, b, g, h = SGD_step(x, z, a, b, c, g, h)
        yield a, b, g, h
