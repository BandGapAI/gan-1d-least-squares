import math


def Phi(x):
    return math.erf(x)


def Phi_plus(x):
    return 1 + Phi(x)


def Phi_minus(x):
    return 1 - Phi(x)
