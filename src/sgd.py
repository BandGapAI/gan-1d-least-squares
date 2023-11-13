from gan import cost


def J_diff_a_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (cost(x, z, a + delta, b, c, g, h) - cost(x, z, a - delta, b, c, g, h))/(2*delta)


def J_diff_b_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (cost(x, z, a, b + delta, c, g, h) - cost(x, z, a, b - delta, c, g, h))/(2*delta)


def J_diff_h_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (cost(x, z, a, b, c, g, h + delta) - cost(x, z, a, b, c, g, h - delta))/(2*delta)


def J_diff_g_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (cost(x, z, a, b, c, g + delta, h) - cost(x, z, a, b, c, g - delta, h))/(2*delta)


def J_diff_h_numerical(x, z, a, b, c, g, h, delta=1e-6):
    return (cost(x, z, a, b, c, g, h + delta) - cost(x, z, a, b, c, g, h - delta))/(2*delta)


def SGD_step(x, z, a0, b0, c, g0, h0, eta):
    a = a0 + eta*J_diff_a_numerical(x, z, a0, b0, c, g0, h0)
    b = b0 + eta*J_diff_b_numerical(x, z, a0, b0, c, g0, h0)
    g = g0 - eta*J_diff_g_numerical(x, z, a0, b0, c, g0, h0)
    h = h0 - eta*J_diff_h_numerical(x, z, a0, b0, c, g0, h0)
    return a, b, g, h


def SGD(x, z, a0, b0, c, g0, h0, eta, stop):
    a, b, g, h = a0, b0, g0, h0
    for _ in range(stop - 1):
        a, b, g, h = SGD_step(x, z, a, b, c, g, h, eta)
        yield a, b, g, h
