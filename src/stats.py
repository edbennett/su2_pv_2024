#!/usr/bin/env python3

import numpy as np


def weighted_mean(results):
    values = np.asarray([result.fit_parameters for result, aic in results])
    weights = np.asarray([np.exp(-aic) for result, aic in results])

    result = list((values * weights[:, np.newaxis]).sum(axis=0) / weights.sum())
    for value in result:
        value.gamma_method()

    return result


def weighted_mean_by_uncertainty(results):
    numerator = sum([result / result.dvalue ** 2 for result in results])
    denominator = sum([1 / result.dvalue ** 2 for result in results])
    mean = numerator / denominator
    mean.gamma_method()
    return mean
