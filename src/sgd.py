from gan import Jgh
from math_utils import Phi_minus, Phi_plus

import numpy as np
import math


def J_diff_b_numerical(x, z, a, b, g, h, delta=1e-6):
    return (Jgh(x, z, a, b + delta, g, h) - Jgh(x, z, a, b - delta, g, h))/(2*delta)


def J_diff_h_numerical(x, z, a, b, g, h, delta=1e-6):
    return (Jgh(x, z, a, b, g, h + delta) - Jgh(x, z, a, b, g, h - delta))/(2*delta)


def J_diff_b(x, z, a, b, g, h):
    eta = a*h + b
    f1 = np.vectorize(lambda x: Phi_plus(a*x + b)*np.exp(-(a*x + b)**2))
    f2 = np.vectorize(lambda z: Phi_minus(a*g*z**2 + eta) *
                      np.exp(-(a*g*z**2 + eta)**2))
    return 1/math.sqrt(math.pi)*np.average(f1(x) - f2(z))


def J_diff_h(x, z, a, b, g, h):
    eta = a*h + b
    f = np.vectorize(lambda z: a*Phi_minus(a*g*z**2 + eta) *
                     np.exp(-(a*g*z**2 + eta)**2))
    return -1/math.sqrt(math.pi)*np.average(f(z))


def _get_parameters(config, b2, h2):
    a, b, c, g, h = [v[1] for v in config.items()]
    b = b2 if b2 is not None else b
    h = h2 if h2 is not None else h
    return a, b, c, g, h


def calc_J_diff(config, x, z, method=J_diff_b, b2=None, h2=None):
    a, b, c, g, h = _get_parameters(config, b2, h2)
    return np.array([[method(x, z, a, b_, g, h_) for b_ in b] for h_ in h])


def SGD_step_bh(x, z, a, b0, g, h0, eps_b, eps_h):
    b = b0 + eps_b*J_diff_b(x, z, a, b0, g, h0)
    h = h0 - eps_h*J_diff_h(x, z, a, b0, g, h0)
    return b, h


def SGD_bh(x, z, a, b0, g, h0, eps_b, eps_h, stop):
    b, h = b0, h0
    for _ in range(stop):
        b, h = SGD_step_bh(x, z, a, b, g, h, eps_b, eps_h)
        yield b, h
