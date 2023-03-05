import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats
import statistics
import math
import numpy as np


def poisson(k, mu):
    for elem in k:
        yield (mu**elem) * (math.e ** (-mu)) / math.gamma(elem + 1)


def ext_half_sum(arr):
    return (arr[0] + arr[-1]) / 2


def quartile(arr, p):
    return arr[math.ceil(arr.size * p) + 1]


def quart_half_sum(arr):
    return (quartile(arr, 0.25) + quartile(arr, 0.75)) / 2


def trimmed_mean(arr):
    summ = 0
    n = arr.size
    for i in range(math.floor(n/4) + 1, math.floor(3 * n / 4) + 1):
        summ += arr[i]
    return 2 * summ / n


def sample_variance(arr):
    summ = 0
    mean = arr.mean()
    for elem in arr:
        summ += (elem - mean) * (elem - mean)
    return summ / arr.size


def characteristics_table(rv, n):
    mean, mean_sq, median, median_sq, ehs, ehs_sq, qhs, qhs_sq, tr, tr_sq, temp = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(1000):
        np.random.seed(i)
        sample = rv.rvs(n)
        temp = sample.mean()
        mean += temp
        mean_sq += temp * temp
        temp = statistics.median(sample)
        median += temp
        median_sq += temp * temp
        temp = ext_half_sum(sample)
        ehs += temp
        ehs_sq += temp * temp
        temp = quart_half_sum(sample)
        qhs += temp
        qhs_sq += temp * temp
        temp = trimmed_mean(sample)
        tr += temp
        tr_sq += temp * temp
    mean /= 1000
    mean_sq /= 1000
    median /= 1000
    median_sq /= 1000
    ehs /= 1000
    ehs_sq /= 1000
    qhs /= 1000
    qhs_sq /= 1000
    tr /= 1000
    tr_sq /= 1000
    data = [["Mean", '{:.3f}'.format(mean), '{:.3f}'.format(mean_sq - mean * mean)],
            ["Median",'{:.3f}'.format(median), '{:.3f}'.format(median_sq - median * median)],
            ["zR", '{:.3f}'.format(ehs), '{:.3f}'.format(ehs_sq - ehs * ehs)],
            ["zQ", '{:.3f}'.format(qhs), '{:.3f}'.format(qhs_sq - qhs * qhs)],
            ["ztr", '{:.3f}'.format(tr), '{:.3f}'.format(tr_sq - tr * tr)]]
    col_names = ["Char", "E(z)", "D(z)"]
    print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))


sns.set()
# np.random.seed(11)
# width = 0.2
n = 100
#
# norm_rv = stats.norm(loc=0, scale=1)
# cauchy_rv = stats.cauchy()
# laplace_rv = stats.laplace()
# pois_rv = stats.poisson(10)
# uniform_rv = stats.uniform(-math.sqrt(3), 2 * math.sqrt(3))
#
# samples = norm_rv.rvs(n)
# samples = cauchy_rv.rvs(n)
# samples = laplace_rv.rvs(n)
# samples = pois_rv.rvs(n)
# samples = uniform_rv.rvs(n)
# samples.sort()

# print(f"n = {n}")
# characteristics_table(uniform_rv, n)

# x = np.linspace(-4, 4, 100)

# sns.histplot(x=samples, stat='density', binwidth=width)  # histogram

# sns.lineplot(x=x, y=norm_rv.pdf(x), color='r')  # distribution graphs
# sns.lineplot(x=x, y=cauchy_rv.pdf(x), color='r')
# sns.lineplot(x=x, y=laplace_rv.pdf(x), color='r')
# sns.lineplot(x=x, y=poisson(x, 10), color='r')
# sns.lineplot(x=x, y=uniform_rv.pdf(x), color='r')

# rv = stats.uniform(-math.sqrt(3), 2*math.sqrt(3))  # boxplot
# df = [rv.rvs(20), rv.rvs(100)]
# sns.boxplot(df, orient='h')

# sns.lineplot(x=x, y=uniform_rv.cdf(x), color='r')  # empirical distribution function
# sns.histplot(x=samples, stat='density', bins=len(samples), element='step', cumulative=True)

# plt.xlim(-4, 4)
# plt.ylim(-0.1, 1.1)

# plt.title(f"n = {n}")
# plt.show()
