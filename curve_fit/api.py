# -*- coding: utf-8 -*-
import abc
from functools import partial
import hashlib
import itertools
from scipy.optimize import minimize

import six
import numpy as np


from pandas import DataFrame
from curve_fit.math import chisquared, calculate_r_squared
from curve_fit.utils import ComputeChiSquare


@six.python_2_unicode_compatible
class Parameter(object):

    """
    Object that representas a parameter.
    """

    def __init__(self, name, initial_value, min_bound=None, max_bound=None, enabled=True):
        """

        :param str name: Name of the parameter
        :param float initial_value: Initial for the parameter
        :param float min_bound: Minimal value for the parameter during the fit
            May be null.
        :param float max_bound: Minimal value for the parameter during the fit
            May be null
        :param bool enabled: If False this parameter will not take part in
            fitting and initial value wil be taken for all function evaluations.
        :return:
        """
        super(Parameter, self).__init__()

        self.name = name
        self.initial_value = initial_value
        self.bounds = (min_bound, max_bound)
        self.enabled=enabled

    def __str__(self):
        return u"<Parameter {}>".format(u", ".join((
            u'name="{}"'.format(self.name),
            u'initial="{}"'.format(self.initial_value),
            u"enabled" if self.enabled else u"disabled",
            u"bounds={}".format(self.bounds) if self.bounds[0] is not None or self.bounds[1] is not None else u"no bounds"
        )))

    __repr__ = __str__


class Parameters(object):

    def __init__(self, parameters=None):
        super(Parameters, self).__init__()
        self.param_list = []
        self.param_map = {}
        self.default_values = {}
        self.param_values = {}
        self.param_stddev = {}
        if parameters is not None:
            for p in parameters:
                self.add_parameter(p)

    def __getitem__(self, item):
        return self.param_map[item]

    @property
    def has_disabled_params(self):
        """
        >>> pars = Parameters([Parameter('a', 1), Parameter('b', 2, enabled=False), Parameter('c', 1)])
        >>> pars.has_disabled_params
        True
        >>> pars = Parameters([Parameter('a', 1)])
        >>> pars.has_disabled_params
        False
        """
        return any([not p.enabled for p in self.param_list])

    @property
    def param_names(self):
        return [p.name for p in self.param_list if p.enabled]

    @property
    def all_param_names(self):
        return [p.name for p in self.param_list]

    def set_values(self, dict):
        self.param_values = dict

    def set_values_from_list(self, param_list):
        self.param_values.clear()
        for p in self.param_list:
            if not p.enabled:
                self.param_values[p.name] = p.initial_value

        self.param_values.update(dict(zip(self.param_names, param_list)))

    def set_stddevs_from_list(self, stddev_list):
        self.param_stddev.clear()
        self.param_stddev = dict(zip(self.param_names, stddev_list))

    def add_parameter(self, parameter):
        self.param_map[parameter.name] = parameter
        self.param_list.append(parameter)

    def get_initial(self, also_disabled=False):
        return [p.initial_value for p in self.param_list if p.enabled or also_disabled]

    def get_bounds(self):
        return [p.bounds for p in self.param_list if p.enabled]

    def param_value_dict_to_list(self, dict):
        return [dict[name] for name in self.param_names]

    def fill_params(self, params):
        """

        >>> pars = Parameters([Parameter('a', 1), Parameter('b', 2, enabled=False), Parameter('c', 1)])
        >>> pars.fill_params([1, 3])
        [1, 2, 3]
        >>> pars = Parameters([Parameter('a', 1), Parameter('b', 2), Parameter('c', 1)])
        >>> pars.fill_params([5, 4, 3])
        [5, 4, 3]

        :param list[float] params: Parameter list, disabled parameters will be
            added to this list.
        :return: Parameters with disabled parameters filled
        :rtype: list[flloat]
        """
        piter = iter(params)
        result = []
        for par in self.param_list:
            if not par.enabled:
                result.append(par.initial_value)
            else:
                result.append(next(piter))
        return result

    def strip_params(self, params):
        """
        >>> pars = Parameters([Parameter('a', 1), Parameter('b', 2, enabled=False), Parameter('c', 1)])
        >>> pars.strip_params([1, 2, 3])
        [1, 3]
        """
        return [params[ii] for ii, p in enumerate(self.param_list) if p.enabled]

    def maybe_strip_params(self, params):
        if len(params) == len(self.all_param_names):
            return self.strip_params(params)
        return params

    def __len__(self):
        return len(self.param_list)

    def __str__(self):
        return u"<Parameters {}>".format(self.param_list)

    __repr__ = __str__


class FitResult(object):

    def __init__(self, fit, fit_data, params, function):
        super(FitResult, self).__init__()
        assert isinstance(fit_data, FitInputData)
        self.fit_data = fit_data
        self.profile = self.fit_y = fit_data.fit_y
        self.geometry = self.fit_x = fit_data.fit_x
        self.params = params
        self.function = function
        self.fit = fit
        """
        :type: :class:`FitPerformer`
        """
        self.__initialize()

    def __initialize(self):
        self.fitted_y = self.fitted_profile = self.function(self.geometry)
        self.plot_geometry = self.plot_x = self.fit_data.plot_x
        self.plot_profile = self.plot_y = self.function(self.plot_geometry)

        # print((self.profile, self.fitted_profile, self.fit_data.fit_sigma_y, len(self.params.param_names)))

        self.chsisq, self.chisq_prob, self.ddof, self.chisq_sum_elems = \
            chisquared(self.profile, self.fitted_profile, self.fit_data.fit_sigma_y, len(self.params.param_names))
        self.rsquared = calculate_r_squared(self.profile, self.fitted_profile)

    def params_as_dataframe(self):
        result = []
        data = self.params.param_values.items()
        data.append(("chisq", self.chsisq))
        data.append(("chisq ddof", self.chsisq/self.ddof))
        data.append(("chisq prob", self.chisq_prob))
        data.append(("fit_to", np.max(self.fit_data.fit_x)))
        data.append(("R squared",  self.rsquared))
        for p, v in data:
            result.append(self.fit.update_dict_for_pandas({
                "param": p,
                "value": v
            }))
        return DataFrame(result)

    def fit_as_dataframe(self, return_plot=False):
        _x = self.fit_data.fit_x
        _y = self.fitted_profile
        if return_plot:
            _x = self.plot_geometry
            _y = self.plot_profile
        data = [self.fit.update_dict_for_pandas({"x": x, "y":y}) for x, y in zip(_x, _y)]
        return DataFrame(data)


class FitInputData(six.with_metaclass(abc.ABCMeta, object)):

    @property
    @abc.abstractmethod
    def fit_x(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def fit_y(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def fit_sigma_y(self):
        return np.ones_like(self.fit_x)

    @property
    def plot_x(self):
        return np.linspace(np.min(self.fit_x), np.max(self.fit_x), 5000)

    @property
    @abc.abstractmethod
    def fit_id(self):
        raise NotImplementedError()

    def update_dict_for_pandas(self, dict):
        sha = self.fit_id
        dict.update({
            "fit_id": sha
        })
        return dict


class SimpleFitInputData(FitInputData):

    def __init__(self, x, y, sigma=None, id=None):
        super(SimpleFitInputData, self).__init__()
        self.x = x
        self.y = y
        self.sigma = sigma
        self.id = id

        assert np.shape(x) == np.shape(y)
        if sigma is not None:
            assert np.shape(sigma) == np.shape(x)

    @property
    def fit_x(self):
        return self.x

    @property
    def fit_y(self):
        return self.y

    @property
    def fit_sigma_y(self):
        if self.sigma is None:
            return super(SimpleFitInputData, self).fit_sigma_y
        return self.sigma

    @property
    def fit_id(self):
        if self.id is None:
            raise ValueError()
        return self.id


class FitPerformer(six.with_metaclass(abc.ABCMeta, object)):


    _IS_READY_FOR_DISABLED_PARAMETERS = False
    """
    parameter.enabled property was added sometime ago, but it needs
    additional care for FitPerformer to work, if subclass doesn't
    set this to True it constructor will raise exceptions if there
    are any parameters set to enabled = True.
    """

    FitResultClass = FitResult

    def __init__(self, fit_data, parameters, fit_name="misc_fit"):
        """

        :param PlotProfile fit_data:
        :return:
        """

        self.fit_data = fit_data
        """
        :type: :class:`FitInputData`
        """

        self.parameters = parameters

        self.__check_disabled_pars()

        self.fit_successful = False
        self.fit_result = None
        """
        :type: :class:`PerformProfileFit`
        """
        self.scipy_fit_result = None
        """
        If this fit method uses scipy minimize it should attach result from
        scipy.
        """

        self.fit_name = fit_name

    @property
    def fit_id(self):
        sha = hashlib.sha256()
        sha.update(self.fit_data.fit_id)
        sha.update(self.fit_name if self.fit_name is not None else "")
        return sha.hexdigest()

    def update_dict_for_pandas(self, dict):
        dict = self.fit_data.update_dict_for_pandas(dict)
        return dict

    @abc.abstractmethod
    def perform_fit_internal(self):
        """
        Performs fit.
        """
        return None

    def _fill_fit_result(self, fitted_function):
        result = self.FitResultClass(self, self.fit_data, self.parameters, fitted_function)
        self.fit_result = result
        return result

    def perform_fit(self):

        self.__check_disabled_pars()

        self.fit_result = None # Piclke will explode if we will not null it now

        result = self.perform_fit_internal()
        self.fit_successful = False
        if not result:
            return

        self.fit_successful = True

        fitted_function = result

        self._fill_fit_result(fitted_function)

        return result

    def __check_disabled_pars(self):
        if self.parameters.has_disabled_params and not self._IS_READY_FOR_DISABLED_PARAMETERS:
            raise ValueError("This fit_performed is not ready for disabled "
                             "parameters, and was instantiated with parameters "
                             "objects from which some are disabled")

class SimpleMinimizeFit(FitPerformer):

    _IS_READY_FOR_DISABLED_PARAMETERS = True

    def _evaluate_function(self, param_list, x):
        return self.fit_func(x, *param_list)

    def __eval_function_wrapper(self, param_list, x):
        param_list = self.parameters.fill_params(param_list)
        return self._evaluate_function(param_list, x)

    def _generate_evaluate_function(self):
        if self.parameters.has_disabled_params:
            return self.__eval_function_wrapper
        else:
            return self._evaluate_function

    def __init__(self, fit_data, parameters, method=None, method_opts = None, fit_name="misc_fit", fit_func=None):
        super(SimpleMinimizeFit, self).__init__(fit_data, parameters, fit_name)
        self.method=method
        self.method_opts = method_opts
        self.fit_func = fit_func

    def _create_chisq_minimize_object(self):
        return ComputeChiSquare(
            self._generate_evaluate_function(),
            self.fit_data.fit_y,
            self.fit_data.fit_sigma_y,
            internal_kwargs={"x": self.fit_data.fit_x}
        )

    def perform_fit_internal(self):
        res = minimize(
            self._create_chisq_minimize_object(),
            self.parameters.get_initial(),
            method=self.method,
            options=self.method_opts,
            bounds=self.parameters.get_bounds())
        self.scipy_fit_result = res
        self.parameters.set_values_from_list(res.x)
        return partial(self._generate_evaluate_function(), res.x)


class OneStageFit(SimpleMinimizeFit):

    _IS_READY_FOR_DISABLED_PARAMETERS = True

    def __init__(self, fit_data, parameters, method=None, method_opts = None, fit_name="misc_fit", initial_parameters = None):
        super(OneStageFit, self).__init__(fit_data, parameters, fit_name=fit_name)
        self.method=method
        self.method_opts = method_opts
        self.initial_parameters = initial_parameters

    def perform_fit_internal(self):
        results = []
        for x0 in self.initial_parameters:
            x0 = self.parameters.maybe_strip_params(x0)
            res = minimize(
                self._create_chisq_minimize_object(),
                x0,
                method=self.method,
                options=self.method_opts,
                bounds=self.parameters.get_bounds())
            results.append(res)
        res = sorted(filter(lambda r: not np.isnan(r.fun), results), key=lambda x: x.fun)[0]
        self.scipy_fit_result = res
        self.parameters.set_values_from_list(res.x)
        return partial(self._evaluate_function, self.parameters.fill_params(res.x))
