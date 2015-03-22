# -*- coding: utf-8 -*-

import abc

import six

import numpy as np

from curve_fit.api import FitPerformer, Parameters, FitResult


@six.add_metaclass(abc.ABCMeta)
class IPartialFit(object):

    @abc.abstractmethod
    def fit_function(self, x, *params):
        pass

    @abc.abstractmethod
    def create_parameters(self):
        pass

    @property
    def parameter_count(self):
        return len(self.parameters.parameter_list)

    def bind_function_with_parameters(self, param_values):
        parameters = self.create_parameters()
        return lambda x: self.fit_function(x, *parameters.param_value_dict_to_list(param_values))

    def create_fit_result(self, original_fit_result, param_values, FitResultClass=FitResult):
        parameters = self.create_parameters()
        parameters.param_values = param_values
        return FitResultClass(
            original_fit_result, original_fit_result.fit_data,
            parameters,
            function=self.bind_function_with_parameters(param_values))

    @classmethod
    def join_parameters(cls, iter):
        param_list = []
        for pf in iter:
            param_list.extend(pf.create_parameters().param_list)

        if len(set(param_list)) != len(param_list):
            raise ValueError("Parameters in partial fit have non-unique names")
        return Parameters(parameters=param_list)

    @classmethod
    def create_fit_function(cls, iter):
        def fit_func(x, *param_list):
            start_param = 0
            result = np.zeros_like(x, dtype=float)

            for pf in iter:
                end_param = start_param+pf.parameter_count
                # print(param_list[start_param:end_param])
                result += pf.fit_function(x, *param_list[start_param:end_param])
                # print(start_param, result)
                start_param=end_param
            return result
        return fit_func



class SimplePartialFit(IPartialFit):

    def __init__(self, params, func):
        super().__init__()
        self.params = params
        self.func = func

    def fit_function(self, x, *params):
        return self.func(x, *params)

    def join_parameters(self):
        return self.params
