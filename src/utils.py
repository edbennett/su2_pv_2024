#!/usr/bin/env python3

import argparse
import itertools
import logging

import numpy as np
import pyerrors as pe


def _backward_enumerate(iterable):
    index = len(iterable) - 1
    for item in iterable:
        yield index, item
        index -= 1


def get_non_none_indices(iterable):
    for start_index, element in enumerate(iterable):
        if element is not None:
            break
    else:
        raise ValueError("dt2E/dt is never not None")

    for end_index, element in _backward_enumerate(iterable):
        if element is not None:
            break

    return start_index, end_index + 1


def partial_corr_mult(array, partial_corr):
    start_index, end_index = get_non_none_indices(partial_corr.content)
    W_middle = (
        array[start_index:end_index]
        * np.asarray(partial_corr.content[start_index:end_index]).T
    ).T
    return pe.Corr(
        [None] * start_index + list(W_middle) + [None] * (len(array) - end_index)
    )


def zip_combinations(*lists, min_count=1):
    max_count = min([len(list_) for list_ in lists])
    if max_count != max([len(list_) for list_ in lists]):
        logging.warning("List lengths are not equal.")

    valid_indices = list(range(0, max_count))
    for count in range(min_count, max_count + 1):
        for selected_indices in itertools.combinations(valid_indices, count):
            yield [[list_[index] for index in selected_indices] for list_ in lists]


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "all":
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, list(map(float, values.split(","))))


def group_params(fit_results, keys):
    params = sorted(set(tuple(result[key] for key in keys) for result in fit_results))
    return {
        param_set: [
            result
            for result in fit_results
            if all([result[key] == param for key, param in zip(keys, param_set)])
        ]
        for param_set in params
    }
