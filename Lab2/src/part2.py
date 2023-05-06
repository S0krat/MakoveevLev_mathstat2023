from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def correct_values():
    points = [-1.8 + i / 5 for i in range(20)]
    c_points = []
    for point in points:
        c_points.append(2 + 2 * point)
    return c_points


def erroneous_values(perturb=False):
    points = [-1.8 + i / 5 for i in range(20)]
    e_points = []
    for point in points:
        e_points.append(2 + 2 * point + stats.norm.rvs(0, 1))
    if perturb:
        e_points[0] += 10
        e_points[19] -= 10

    return e_points


def lsm_function(params, e_values):
    points = [-1.8 + i / 5 for i in range(20)]
    values = [params[0] * p + params[1] for p in points]
    sum_of_squares = 0
    for i in range(20):
        sum_of_squares += (values[i] - e_values[i]) ** 2
    return sum_of_squares


def lmm_function(params, e_values):
    points = [-1.8 + i / 5 for i in range(20)]
    values = [params[0] * p + params[1] for p in points]
    sum_of_modules = 0
    for i in range(20):
        sum_of_modules += abs(values[i] - e_values[i])
    return sum_of_modules


def opti():
    points = [-1.8 + i / 5 for i in range(20)]
    c_values = correct_values()
    e_values = erroneous_values(True)
    par = np.array([0, 0])
    res = minimize(lsm_function, par, args=e_values, method='nelder-mead')
    print("Метод наименьших квадратов:", res.x)
    lsm_values = [res.x[0] * p + res.x[1] for p in points]
    lsm_distance = max([abs(lsm_values[i] - c_values[i]) for i in range(20)])
    print("max dist = ", lsm_distance)
    res = minimize(lmm_function, par, args=e_values, method='nelder-mead')
    print("Метод наименьших модулей:", res.x)
    lmm_values = [res.x[0] * p + res.x[1] for p in points]
    lmm_distance = max([abs(lmm_values[i] - c_values[i]) for i in range(20)])
    print("max dist = ", lmm_distance)
    plt.grid()
    plt.scatter(points, e_values, label='Выборка')
    plt.plot(points, c_values, label='Модель')
    plt.plot(points, lsm_values, label='Метод наименьших квадратов')
    plt.plot(points, lmm_values, label='Метод наименьших модулей')
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.show()
