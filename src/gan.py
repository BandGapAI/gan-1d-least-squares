import numpy as np


def Discriminator(x, a, b, c, g, h):
    return np.vectorize(lambda x: 1.0/(1.0 + a*np.exp(-b*x)))(x)


def Generator(z, g, h):
    return np.vectorize(lambda z: g*(z**2) + h)(z)


def cost(x, z, a, b, c, g, h):
    return np.average(np.square(Discriminator(x, a, b, c, g, h))) + \
        np.average(
            np.square(1 - Discriminator(Generator(z, g, h), a, b, c, g, h)))


def sample_xz(config, n_samples=1000):
    c = config['c']
    x = np.random.exponential(1/c, n_samples)
    z = np.random.rayleigh(1/np.sqrt(2), n_samples)

    return x, z
