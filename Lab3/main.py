import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sample(sample):
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.show()


def plot_sample_w_error(sample, epsilon):
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    mode = sample_mode(sample, epsilon)
    plt.axhline(mode[0], color='r', alpha=0.5)
    plt.axhline(mode[1], color='r', alpha=0.5)
    red_n = []
    red_sample_high = []
    red_sample_low = []
    for i in range(200):
        if sample_high[i] > mode[1] and sample_low[i] < mode[0]:
            red_n.append(i + 1)
            red_sample_high.append(sample_high[i])
            red_sample_low.append(sample_low[i])
    plt.fill_between(red_n, red_sample_low, red_sample_high, color='red', alpha=0.5)
    plt.show()


def sample_estimates(sample, epsilon):
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    print([np.min(sample_low), np.max(sample_high)])


def occurrence_intervals(sample, epsilon):
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    borders = np.concatenate([sample_high, sample_low])
    borders = np.sort(borders)
    intervals = np.array([[borders[i], borders[i + 1]] for i in range(borders.size - 1)])
    occurrences = []
    for interval in intervals:
        n = 0
        for num in sample:
            if num - epsilon <= interval[0] and num + epsilon >= interval[1]:
                n += 1
        occurrences.append(n)
    return intervals, occurrences


def mode_intervals(intervals, occurrences):
    occ_max = max(occurrences)
    max_intervals = [i for n, i in enumerate(intervals) if occurrences[n] == occ_max]
    borders = np.concatenate(max_intervals)
    # print("Макисмальная клика:", occ_max)
    # print("Интервалы моды:")
    # print(max_intervals)
    return [np.min(borders), np.max(borders)]


def sample_mode(sample, epsilon):
    intervals, occurrences = occurrence_intervals(sample, epsilon)
    return mode_intervals(intervals, occurrences)


def frequency_graph(sample, epsilon):
    intervals, occurrences = occurrence_intervals(sample, epsilon)
    for interval, occurrence in zip(intervals, occurrences):
        plt.plot(interval, [occurrence, occurrence], color='blue')
    plt.xlabel('mV')
    plt.ylabel('freq')
    mode = mode_intervals(intervals, occurrences)
    plt.plot([mode[0], mode[0]], [0, max(occurrences)], 'r--')
    plt.plot([mode[1], mode[1]], [0, max(occurrences)], 'r--')
    plt.show()


def estimate_uncertainty_center(sample, epsilon):
    k = (sample[-1] + sample[0]) / 2
    w = (sample[-1] - sample[0]) / (2 * epsilon)
    print('oskorbin_center_k = ', k, '\nw = ', w)
    epsilon = (sample[-1] - sample[0]) / 2
    n = range(1, len(sample) + 1)
    plt.plot(n, sample)
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    plt.plot(n, sample_high, '--', color='blue', alpha=0.5)
    plt.plot(n, sample_low, '--', color='blue', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.fill_between(n, sample_low, sample_high, alpha=0.5)
    plt.axhline(k, color='r', alpha=0.5)
    plt.show()


def jaccard_index(sample, epsilon):
    sample_low = sample - epsilon
    sample_high = sample + epsilon
    return (sample_high[0] - sample_low[-1]) / (sample_high[-1] - sample_low[0])


def relative_mode_width(sample, epsilon):
    mode = sample_mode(sample, epsilon)
    return (mode[1] - mode[0]) / (sample[0] + sample[-1] + 2 * epsilon)


df = pd.read_excel('data1.xlsx')
print(df.head(10))
# data = np.array(df['mV'])
# eps = 10 ** (-3)
# print(jaccard_index(data, eps))
# print(relative_mode_width(data, eps))

