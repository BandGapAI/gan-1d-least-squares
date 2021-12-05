# Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks using Error Function Integrals

This is a repository that contains numerical simulations for the paper
__Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial
Networks using Error Function Integrals__ (Graham W. Pulford, Kirill Kondrashov).

The code is related to stochastic gradient descent (SGD) method, the analytical
part is described in the paper and not present in the code. We provide a comparison
of both analytical and MC-based computations for two special cases.

To run the code, open the `.ipynb` file in Jupyter notebook environment. The
requirements are installed at the first executional cell. You need Python
3.x to run the code.

## Visualizations 
The graphs for the executed code are shown below. Extended comments
on them can be found in the article.

Case A: Single SDG run.

![case a single](content/case_a_surface_single_run.png)

Case A: comparison between analytical and MC results.

![case a plot](content/case_a.png)

Case B: Single SDG run.

![case b single](content/case_b_surface_single_run.png)

Case B: comparison between analytical and MC results.

![case b plot](content/case_b.png)
