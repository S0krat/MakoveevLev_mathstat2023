import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import pandas as pd
from itertools import combinations


def plot_sample():
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.show()


def plot_sample_w_error():
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    coeffs = [1.082e-02, 5.827e-05]
    plt.plot([0, 200], [coeffs[0], coeffs[0] + 200 * coeffs[1]], color='red')
    plt.show()


def var_exp():
    obj = [1 for _ in range(200)] + [0, 0]
    lhs_ineq = []
    for i in range(200):
        lhs_ineq.append([0 for _ in range(i)] + [-epsilon] + [0 for _ in range(i + 1, 200)] + [-1, -i - 1])
    for i in range(200):
        lhs_ineq.append([0 for _ in range(i)] + [-epsilon] + [0 for _ in range(i + 1, 200)] + [1, i + 1])
    rhs_ineq = [-sample[i] for i in range(200)] + [sample[i] for i in range(200)]
    bnd = [(1, float("inf")) for _ in range(200)] + [(-5, 5), (-5, 5)]
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="highs")
    return opt.x[200:]


def var_exp_narrow():
    obj = [1 for _ in range(200)] + [0, 0]
    lhs_ineq = []
    for i in range(200):
        lhs_ineq.append([0 for _ in range(i)] + [-epsilon] + [0 for _ in range(i + 1, 200)] + [-1, -i - 1])
    for i in range(200):
        lhs_ineq.append([0 for _ in range(i)] + [-epsilon] + [0 for _ in range(i + 1, 200)] + [1, i + 1])
    rhs_ineq = [-sample[i] for i in range(200)] + [sample[i] for i in range(200)]
    bnd = [(0, float("inf")) for _ in range(200)] + [(-5, 5), (-5, 5)]
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="highs")
    print(opt)
    return opt.x


def plot_optimized():
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    coefs = var_exp_narrow()
    plt.plot([0, 200], [coefs[200], coefs[200] + 200 * coefs[201]])
    new_sample_high = []
    new_sample_low = []
    for i in range(200):
        new_sample_high.append(sample[i] + epsilon * coefs[i])
        new_sample_low.append(sample[i] - epsilon * coefs[i])
    plt.fill_between(n, new_sample_low, new_sample_high, alpha=0.5, color='yellow')
    plt.show()


def plot_coeffs():
    n = range(1, len(sample) + 1)
    coeffs = var_exp_narrow()[:200]
    plt.plot(n, coeffs)
    plt.plot([1, 200], [1, 1])
    plt.xlabel('n')
    plt.ylabel('w')
    plt.show()


def plot_regress_res_1():
    n = range(1, len(sample) + 1)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    new_sample = []
    coeffs = var_exp()
    for i in range(1, 201):
        value = coeffs[0] + i * coeffs[1]
        sample_low[i-1] -= value
        sample_high[i-1] -= value
        new_sample.append(sample[i-1] - value)
    plt.plot(n, new_sample)
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    plt.plot([1, 200], [0, 0], color='red')
    plt.show()


def plot_regress_res_2():
    n = range(1, len(sample) + 1)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    new_sample = []
    coeffs = var_exp_narrow()[200:]
    for i in range(1, 201):
        value = coeffs[0] + i * coeffs[1]
        sample_low[i-1] -= value
        sample_high[i-1] -= value
        new_sample.append(sample[i-1] - value)
    plt.plot(n, new_sample)
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    plt.plot([1, 200], [0, 0], color='red')
    plt.show()


def information_nodes():
    n = range(1, 201)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    all_bounds = np.concatenate([sample_high, sample_low])
    nodes = []
    for i, j in combinations(range(400), 2):
        x_1 = i % 200 + 1
        x_2 = j % 200 + 1
        if x_1 == x_2:
            continue
        beta_1 = (all_bounds[i] - all_bounds[j]) / (x_1 - x_2)
        beta_0 = all_bounds[j] - beta_1 * x_2
        is_break = False
        for k in n:
            if not sample_low[k - 1] - 0.00001 <= beta_0 + beta_1 * k <= sample_high[k - 1] + 0.00001:
                is_break = True
                break
        if not is_break:
            nodes.append([beta_0, beta_1])
    return nodes


def information_set():
    points = np.array(information_nodes())
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    max_0 = max(points[:, 0])
    max_1 = max(points[:, 1])
    min_0 = min(points[:, 0])
    min_1 = min(points[:, 1])
    print([min_0, max_0])
    print([min_1, max_1])
    plt.plot([min_0, min_0], [min_1, max_1], 'r-')
    plt.plot([max_0, max_0], [min_1, max_1], 'r-')
    plt.plot([min_0, max_0], [min_1, min_1], 'r-')
    plt.plot([min_0, max_0], [max_1, max_1], 'r-')
    plt.xlabel('$\\beta_0$')
    plt.ylabel('$\\beta_1$')
    plt.show()


def joint_dep_corridor():
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    corr_low = []
    corr_high = []
    nodes = information_nodes()
    for i in n:
        min_value = max_value = nodes[0][0] + nodes[0][1] * i
        for node in nodes:
            value = node[0] + node[1] * i
            if value < min_value:
                min_value = value
            if value > max_value:
                max_value = value
        corr_high.append(max_value)
        corr_low.append(min_value)
    plt.fill_between(n, corr_low, corr_high, alpha=0.5, color='red')
    # right_extra_high = [corr_high[-1] + (corr_high[-1] - corr_high[-2]) * i for i in range(1, 51)]
    # right_extra_low = [corr_low[-1] + (corr_low[-1] - corr_low[-2]) * i for i in range(1, 51)]
    # plt.fill_between(range(201, 251), right_extra_low, right_extra_high, alpha=0.5, color='red')
    # left_extra_high = [corr_high[0] + (corr_high[0] - corr_high[1]) * i for i in range(1, 51)]
    # left_extra_low = [corr_low[0] + (corr_low[0] - corr_low[1]) * i for i in range(1, 51)]
    # left_extra_high.reverse()
    # left_extra_low.reverse()
    # plt.fill_between(range(-49, 1), left_extra_low, left_extra_high, alpha=0.5, color='red')
    # plt.grid()
    plt.show()


df = pd.read_excel('data1.xlsx')
sample = np.array(df['mV'])
epsilon = 10 ** (-3)
information_set()

