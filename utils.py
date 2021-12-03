import os
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

def load_analytical(fname, param):
  """
  Utility functions to compare with analytical implementation
  """
  if not os.path.isfile(fname):
      return None
  return loadmat(fname)[param].transpose()

def _set_subplot(title):
  plt.title(title)
  plt.xlabel('B')
  plt.ylabel('H')

def show_comparative(b_paths, h_paths, thetas):
  """
  Show comparison between analytical results and SGD
  """
  fig = plt.subplots(4, 2, figsize=(15, 15))

  ax_sgd_traj = plt.subplot(2,2,1)
  for i in range(len(b_paths)):
    plt.plot(b_paths[i], h_paths[i], color='tab:blue')
  _set_subplot('SGD Trajectories')

  ax_sgd_endpoints = plt.subplot(2,2,2)
  for i in range(len(b_paths)):
    plt.scatter(b_paths[i][-1], h_paths[i][-1], color='tab:blue')
  _set_subplot('SGD End Points')

  if thetas is None:
    return
  
  ax_analytical_traj = plt.subplot(2,2,3)
  for i in range(thetas.shape[0]):
    plt.plot(thetas[i][1], thetas[i][3], color='tab:orange')
  _set_subplot('Analytical Trajectories')

  ax_analytical_endpoints = plt.subplot(2,2,4)
  for i in range(thetas.shape[0]):
    plt.scatter(thetas[i][1][-1], thetas[i][3][-1], color='tab:orange')
  _set_subplot('Analytical End Points')
  ax_analytical_endpoints.set_ylim(ax_sgd_endpoints.get_ylim())
  ax_analytical_endpoints.set_xlim(ax_sgd_endpoints.get_xlim())

  plt.show()
