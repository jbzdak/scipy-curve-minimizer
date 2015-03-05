# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import chisqprob, linregress


def chisquared(
        observed_series,
        theoretical_series,
        observed_series_errors,
        parameters_used_in_fit = 0):

    """
    **Tests**

    Taken from: Taylor: 'Introduction to Error Analysis' second edition.

    >>> obs = [8, 10, 16, 6]
    >>> expected = [6.4, 13.6, 13.6, 6.4]
    >>> sigma = np.sqrt(expected)
    >>> sum, prob, __, __ = chisquared(obs, expected, sigma, 0)
    >>> "{:.2f}".format(sum)
    '1.80'
    >>> "{:.2f}".format(prob)
    '0.61'

    :param observed_series: Series of observed datapoints
    :type observed_series: array-like
    :param theoretical_series: Tcalculate_r_squaredheoretical distribution of data points
    :type theoretical_series: array-like
    :param observed_series_errors: Errors in observed data points in each bin
    :type observed_series_errors: array-like
    :param int parameters_used_in_fit:
    :return:
    """

    if (len(observed_series) != len(theoretical_series) or
        len(theoretical_series) != len(observed_series_errors)):
        raise ValueError()

    bins = len(observed_series_errors)

    theoretical_series = np.asanyarray(theoretical_series)

    parts = np.power((theoretical_series - observed_series) / observed_series_errors, 2)

    chisq = np.sum(parts)
    ddof = bins-1-parameters_used_in_fit
    return chisq, chisqprob(chisq, ddof), ddof, parts



def calculate_r_squared(observed, predicted):
    __, __, r_value, __, __ =  linregress(observed, predicted)
    return r_value