import scipy.stats as stats
import numpy as np
from statistics import mean, variance
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def bivariate_sampling(size, rho):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)


def mixture_of_normal_distributions(size):
    return 0.9 * bivariate_sampling(size, 0.9) + 0.1 * bivariate_sampling(size, -0.9)


def quadrant_cc(x, y):
    n = [0, 0, 0, 0]
    size = len(x)
    x_med = np.median(x)
    y_med = np.median(y)
    for i in range(size):
        if x[i] >= x_med and y[i] >= y_med:
            n[0] += 1
        elif x[i] < x_med and y[i] >= y_med:
            n[1] += 1
        elif x[i] < x_med and y[i] < y_med:
            n[2] += 1
        else:
            n[3] += 1
    return (n[0] + n[2] - n[1] - n[3]) / size


def print_table(size):
    table = """\\begin{table}[h!]
\\centering
\\begin{tabular}{|c||c|c|c||c|c|c||c|c|c|}
\\hline
$\\rho$ & $E(r_P)$ & $E(r_P^2)$ & $D(r_P)$ & $E(r_S)$ & $E(r_S^2)$ & $D(r_S)$ & $E(r_Q)$ & $E(r_Q^2)$ & $D(r_Q)$\\\\
\\hline
"""
    for rho in [0, 0.5, 0.9]:
        pearson_values = []
        spearman_values = []
        quadrant_values = []
        for i in range(1000):
            sample = bivariate_sampling(size, rho)
            xs, ys = sample[:, 0], sample[:, 1]
            pearson_values.append(stats.pearsonr(xs, ys)[0])
            spearman_values.append(stats.spearmanr(xs, ys)[0])
            quadrant_values.append(quadrant_cc(xs, ys))
        line = str(rho)
        for values in [pearson_values, spearman_values, quadrant_values]:
            mean_value = np.around(mean(values), decimals=3)
            mean_sq_value = np.around(sum([elem ** 2 for elem in values]) / 1000, decimals=3)
            var_value = np.around(variance(values), decimals=3)
            line += f" & {mean_value} & {mean_sq_value} & {var_value}"
        line += " \\\\ \n"
        table += line
    pearson_values = []
    spearman_values = []
    quadrant_values = []
    for i in range(1000):
        sample = mixture_of_normal_distributions(size)
        xs, ys = sample[:, 0], sample[:, 1]
        pearson_values.append(stats.pearsonr(xs, ys)[0])
        spearman_values.append(stats.spearmanr(xs, ys)[0])
        quadrant_values.append(quadrant_cc(xs, ys))
    line = "MIX"
    for values in [pearson_values, spearman_values, quadrant_values]:
        mean_value = np.around(mean(values), decimals=3)
        mean_sq_value = np.around(sum([elem ** 2 for elem in values]) / 1000, decimals=3)
        var_value = np.around(variance(values), decimals=3)
        line += f" & {mean_value} & {mean_sq_value} & {var_value}"
    line += " \\\\ \n"
    table += line
    table += """\\hline
\\end{tabular}
\\caption{size=}
\\label{tab:size}
\\end{table}"""
    print(table)


def scattering_ellipse(x, y, axes, title):
    n_std = 3
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0] * 2 * n_std, height=lambda_[1] * 2 * n_std,
                  angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor='black')
    ell.set_facecolor('none')
    axes.grid()
    axes.add_patch(ell)
    axes.scatter(x, y)
    axes.set_title(title)


def plot_ellipses(size):
    _, ax = plt.subplots(1, 3)
    rho = [0, 0.5, 0.9]
    for i in range(3):
        sample = bivariate_sampling(size, rho[i])
        xs, ys = sample[:, 0], sample[:, 1]
        scattering_ellipse(xs, ys, ax[i], f"$\\rho$ = {rho[i]}")
    plt.show()