import numpy as np


def Discriminator(x, a, b, c, g, h):
    return np.vectorize(lambda x: 1.0/(1.0 + 1.0/g/c * np.exp(-(c - 1.0/g)*x - h/g)))(x)


def Generator(z, g, h):
    return np.vectorize(lambda z: g*(z**2) + h)(z)


def Jgh(x, z, a, b, c, g, h):
    return np.average(np.square(Discriminator(x, a, b, c, g, h))) + \
        np.average(
            np.square(1 - Discriminator(Generator(z, g, h), a, b, c, g, h)))


def sample_xz(config, n_samples=1000):
    a, b, c, g, h = [v[1] for v in config.items()]
    x = np.random.exponential(1/c, n_samples)
    z = np.random.rayleigh(1/np.sqrt(2), n_samples)

    return x, z
