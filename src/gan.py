import numpy as np
from math_utils import Phi


def Discriminator(x, a, b):
  return np.vectorize(lambda x: 0.5*(1 + Phi(a*x + b)))(x)


def Generator(z, g, h):
  return np.vectorize(lambda z: g*(z**2) + h)(z)


def Jbh(x, z, a, b, g, h):
  return np.average(np.square(Discriminator(x, a, b))) + \
      np.average(np.square(1 - Discriminator(Generator(z, g, h), a, b)))


def sample_Jbh(config, n_samples=1000):
  a, b, c, g, h = [v[1] for v in config.items()]
  x = np.random.exponential(1/c, n_samples)
  z = np.random.rayleigh(1/np.sqrt(2), n_samples)
  bg, hg = np.meshgrid(b, h)
  jbh = np.array([[Jbh(x, z, a, b_, g, h_) for b_ in b] for h_ in h])

  return jbh, bg, hg, x, z
