# -*- coding: utf-8 -*-
from functools import partial
import numpy as np


def fix_sigma(sigma):
    sigma[sigma == 0.0] = np.min(sigma[sigma != 0.0])
    return sigma


class BaseMinimizerFunction(object):

    def __init__(self, internal_function, internal_kwargs=None):
        self._internal_function = internal_function
        if internal_kwargs is None :
            internal_kwargs = {}
        self.internal_kwargs = internal_kwargs


class ComputeChiSquare(BaseMinimizerFunction):

    """
    This class computes chi-square/ddof test for goodnes of fit (value
    that should be minimized).

    >>> x_array = np.asanyarray([1.0, 2.0, 3.0])
    >>> func = lambda a: (a*x_array)+1

    First check it without sigma

    >>> c = ComputeChiSquare(func, func(2), None, 0)

    Series are the same so chi should be 0

    >>> c([2])
    0.0

    >>> c([1])# doctest:+ELLIPSIS
    4.6666...

    >>> c = ComputeChiSquare(func, func(2), [1, 1E6, 1E6], 0)

    >>> c([2])
    0.0

    >>> c([1])# doctest:+ELLIPSIS
    0.3333...

    >>> c = ComputeChiSquare(func, func(2), [1E6, 1E6, 1], 0)
    >>> c([2])
    0.0

    >>> c([1])# doctest:+ELLIPSIS
    3.00...
    """

    def __init__(self, internal_function, expected, sigma_y=None, ddof=0, internal_kwargs=None):
        super(ComputeChiSquare, self).__init__(internal_function, internal_kwargs)
        self.__expected = expected

        if sigma_y is None:
            sigma_y = np.ones_like(expected)

        self.__sigma_y = fix_sigma(np.asarray(sigma_y, dtype=float))

        self.__ddof = len(expected) - ddof

    def __call__(self, args):
        """
        >>> function = lambda  a, x: a[0]*x + a[1]

        >>> cc = ComputeChiSquare(function, np.ones(10)*10,
        ...         internal_kwargs={'x': np.arange(10)})
        >>> cc([0, 10])
        0.0
        >>> cc([0, 11])
        1.0
        >>> cc([0, 9])
        1.0
        >>> cc = ComputeChiSquare(function, np.ones(10)*10, sigma_y=np.ones(10)*0.5,
        ...         internal_kwargs={'x': np.arange(10)})
        >>> cc([0, 10])
        0.0
        >>> cc([0, 11])
        4.0
        >>> cc([0, 9])
        4.0

        :param args:
        :return:
        """
        modelled = self._internal_function(args, **self.internal_kwargs)
        return float(np.sum(np.power((modelled - self.__expected)/self.__sigma_y, 2))/self.__ddof)
