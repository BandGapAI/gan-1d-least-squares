import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def show_paths(g_paths, h_paths):
    """
    Show path in b/h parameter space
    """
    g_paths, h_paths = np.array(g_paths), np.array(h_paths)

    plt.figure(figsize=(10, 10))

    mean_g = np.mean(g_paths, axis=0)
    std_g = np.std(g_paths, axis=0)
    mean_h = np.mean(h_paths, axis=0)
    std_h = np.std(h_paths, axis=0)
    plt.plot(mean_g, label=r'$\bar g \pm 3\sigma$')
    plt.fill_between(range(len(mean_g)), mean_g - 3 *
                     std_g, mean_g + 3*std_g, alpha=0.7)
    plt.plot(mean_h, label=r'$\bar h \pm 3\sigma$')
    plt.fill_between(range(len(mean_h)), mean_h - 3 *
                     std_h, mean_h + 3*std_h, alpha=0.7)
    plt.legend()
    plt.show()
    print('solution g = %1.4f +/- %1.4f, h = %1.4f +/- %1.4f' %
          (mean_g[-1], 3*std_g[-1], mean_h[-1], 3*std_h[-1]))


def create_3d_plot():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=45, elev=20)
    ax.invert_xaxis()
    return fig, ax


def load_analytical(fname, param):
    """
    Utility functions to compare with analytical implementation
    """
    if not os.path.isfile(fname):
        return None
    return loadmat(fname)[param].transpose()


def _set_subplot(title):
    plt.title(title)
    plt.xlabel('G')
    plt.ylabel('H')


def show_comparative(g_paths, h_paths, thetas):
    # FIXME:
    """
    Show comparison between analytical results and SGD
    """
    plt.subplots(1, 2, figsize=(15, 7))

    # plt.subplot(2, 2, 1)
    plt.subplot(1, 2, 1)
    for i in range(len(g_paths)):
        plt.plot(g_paths[i], h_paths[i], color='tab:blue')
    _set_subplot('SGD Trajectories')

    # ax_sgd_endpoints = plt.subplot(2, 2, 2)
    ax_sgd_endpoints = plt.subplot(1, 2, 2)
    for i in range(len(g_paths)):
        plt.scatter(g_paths[i][-1], h_paths[i][-1], color='tab:blue')
    _set_subplot('SGD End Points')

    # if thetas is None:
    #     return

    # plt.subplot(2, 2, 3)
    # for i in range(thetas.shape[0]):
    #     plt.plot(thetas[i][1], thetas[i][3], color='tab:orange')
    # _set_subplot('Analytical Trajectories')

    # ax_analytical_endpoints = plt.subplot(2, 2, 4)
    # for i in range(thetas.shape[0]):
    #     plt.scatter(thetas[i][1][-1], thetas[i][3][-1], color='tab:orange')
    # _set_subplot('Analytical End Points')
    # ax_analytical_endpoints.set_ylim(ax_sgd_endpoints.get_ylim())
    # ax_analytical_endpoints.set_xlim(ax_sgd_endpoints.get_xlim())

    plt.show()
