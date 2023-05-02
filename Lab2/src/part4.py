from part3 import maximum_likelihood_method
import numpy as np
from scipy import stats


def max_likelihood_interval(sample):
    params = maximum_likelihood_method(sample)
    border = stats.norm.ppf(loc=params[0], scale=params[1], q=0.975)
    return [-params[0] - border, border]


def student_interval(sample):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample)
    return [mean - std * (stats.t.ppf(0.975, n - 1)) / np.sqrt(n - 1),
            mean + std * (stats.t.ppf(0.975, n - 1)) / np.sqrt(n - 1)]


def chi2_interval(sample):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample)
    return [std * np.sqrt(n) / np.sqrt(stats.chi2.ppf(0.975, n-1)),
            std * np.sqrt(n) / np.sqrt(stats.chi2.ppf(0.025, n-1))]


def get_intervals():
    for size in [20, 100]:
        print(f"\nsize = {size}")
        sample = stats.norm.rvs(size=size)
        print("Среднее и выборочное стд отклонение выборки:")
        print([np.mean(sample), np.std(sample)])
        print("Интервальная оценка макс. правдоподобия:")
        print(max_likelihood_interval(sample))
        print("Доверительный интервал мат. ожидания Стьюдента:")
        print(student_interval(sample))
        print("Доверительный интервал стд отклонения кси квадрат:")
        print(chi2_interval(sample))
