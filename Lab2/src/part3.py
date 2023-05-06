import math
from scipy import stats
from scipy.optimize import minimize
import numpy as np


def likelihood_function(params, sample):
    n = len(sample)
    cf = 1 / ((2 * math.pi * (params[1] ** 2)) ** (n / 2.))
    s = sum([(x - params[0]) ** 2 for x in sample])
    return -cf * math.exp(-s / (2 * (params[1] ** 2)))


def maximum_likelihood_method(sample):
    params = np.array([0, 1])
    res = minimize(likelihood_function, params, args=sample, method='nelder-mead')
    return res.x


def chi2(sample):
    n = len(sample)
    k = int(1.72 * (n ** (1 / 3.)))
    l_border = min(sample)
    gap_length = (max(sample) - l_border) / k
    gap_boundaries = [l_border + i * gap_length for i in range(k + 1)]

    prob = [0 for _ in range(k)]
    prob[0] = stats.norm.cdf(gap_boundaries[1])
    for i in range(1, k - 1):
        prob[i] = stats.norm.cdf(gap_boundaries[i + 1]) - stats.norm.cdf(gap_boundaries[i])
    prob[-1] = 1 - stats.norm.cdf(gap_boundaries[-2])

    nums = [0 for _ in range(k)]
    for elem in sample:
        for i in range(k):
            if gap_boundaries[i] <= elem < gap_boundaries[i + 1]:
                nums[i] += 1
                break

    chi_sq = 0
    for i in range(k):
        chi_sq += ((nums[i] - n * prob[i]) ** 2) / (n * prob[i])

    print("k = ", k)
    print("Границы интервалов: ", gap_boundaries)
    print("Вероятности: ", prob)
    print("Количество: ", nums)
    print("Хи квадрат распределение:", stats.chi2.ppf(0.95, k - 1))
    print("Получилось:", chi_sq)


def chi2_method():
    print("Нормальное распределение:")
    sample = stats.norm.rvs(size=100)
    chi2(sample)
    print("Распределение Лапласа:")
    sample = stats.laplace.rvs(size=100)
    chi2(sample)
    print("Равномерное распределение:")
    sample = stats.uniform.rvs(loc=-2, scale=4, size=100)
    chi2(sample)
