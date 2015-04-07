import importlib
import numpy as np
from types import FunctionType
import inspect

import sklearn.base

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_nd_to_sa, np_sa_to_nd


class WrapSKLearnException(Exception):
    pass


def unpickle_estimator(sk_cls, params):
    """ method used by pickle to unpickle wrapped estimators. Do not call
    directly """
    cls = __wrap_estimator(sk_cls)
    return cls(**params)


def __wrap_estimator(sk_cls):
    """Wraps a scikit BaseEstimator class inside a UPSG Stage and returns it
    """
    class WrappedEstimator(RunnableStage):
        __sk_cls = sk_cls

        def __init__(self, **kwargs):
            self.__sk_instance = None
            self.__params = kwargs
            self.__cached_uos = {}
            self.__sk_instance = None
            self.__fitted = False

        def __reduce__(self):
            return (unpickle_estimator, (self.__sk_cls, self.__params))

        def __uo_to_np(self, uo):
            try:
                (A, dtype) = self.__cached_uos[uo]
            except KeyError:
                A_sa = uo.to_np()
                A, dtype = np_sa_to_nd(A_sa)
                self.__cached_uos[uo] = (A, dtype)
            return (A, dtype)

        def __np_to_uo(self, A, dtype=None):
            A_sa = np_nd_to_sa(A, dtype)
            uo_out = UObject(UObjectPhase.Write)
            uo_out.from_np(A_sa)
            return uo_out

        __input_keys = set()
        __output_keys = set()
        __funcs_to_run = {}

        __input_keys.add('params_in')

        __output_keys.add('params_out')

        def __do_get_params(self, **kwargs):
            uo = UObject(UObjectPhase.Write)
            uo.from_dict(self.get_params())
            return uo
        __funcs_to_run['params_out'] = __do_get_params

        # It would be nicer to use class hierarchy than hasattr, but sklearn
        # doesn't put everything in interfaces
        if hasattr(sk_cls, 'fit_transform'):
            __input_keys.add('X_train')
            __input_keys.add('y_train')
            __input_keys.add('fit_params')
            __output_keys.add('X_new')

            def __do_fit_transform(self, **kwargs):
                (X_train, X_train_dtype) = self.__uo_to_np(kwargs['X_train'])
                try:
                    (y_train, y_train_dtype) = (
                        self.__uo_to_np(kwargs['y_train']))
                except KeyError:
                    y_train = None
                # TODO test this part
                try:
                    fit_params = kwargs['fit_params'].to_dict()
                except KeyError:
                    fit_params = {}
                fit_params = {}
                X_new_nd = self.__sk_instance.fit_transform(X_train, y_train,
                                                            **fit_params)
                return self.__np_to_uo(X_new_nd, X_train_dtype)
            __funcs_to_run['X_new'] = __do_fit_transform
        if hasattr(sk_cls, 'fit'):
            __input_keys.add('X_train')
            __input_keys.add('y_train')
            __input_keys.add('sample_weight')

            def __fit(self, **kwargs):
                if self.__fitted:
                    return
                (X_train, X_train_dtype) = self.__uo_to_np(kwargs['X_train'])
                (y_train, y_train_dtype) = self.__uo_to_np(kwargs['y_train'])
                try:
                    (sample_weight, sample_weight_dtype) = self.__uo_to_np(
                        kwargs['sample_weight'])
                except KeyError:
                    sample_weight = None
                self.__sk_instance.fit(X_train, y_train,
                                       sample_weight)
                self.__fitted = True
            if hasattr(sk_cls, 'score'):
                __output_keys.add('score')
                __input_keys.add('X_test')
                __input_keys.add('y_test')

                def __do_score(self, **kwargs):
                    self.__fit(**kwargs)
                    (X_test, X_test_dtype) = self.__uo_to_np(kwargs['X_test'])
                    (y_test, y_test_dtype) = self.__uo_to_np(kwargs['y_test'])
                    try:
                        sample_weight, sample_weight_dtype = self.__uo_to_np(
                            kwargs['sample_weight'])
                    except KeyError:
                        sample_weight = None
                    score = self.__sk_instance.score(X_test, y_test,
                                                     sample_weight)
                    return self.__np_to_uo(score)
                __funcs_to_run['score'] = __do_score
            # TODO these share a lot of code, but I ran into a lot of
            #   scoping issues trying to factor them out.
            #   (http://stackoverflow.com/questions/9505979/the-scope-of-names-defined-in-class-block-doesnt-extend-to-the-methods-blocks)
            #   Find a better way to make this prettier
            if hasattr(sk_cls, 'predict'):
                __output_keys.add('y_pred')
                __input_keys.add('X_test')

                def __do_predict(self, **kwargs):
                    self.__fit(**kwargs)
                    (X_test, X_test_dtype) = self.__uo_to_np(kwargs['X_test'])
                    result = self.__sk_instance.predict(X_test)
                    return self.__np_to_uo(result)
                __funcs_to_run['y_pred'] = __do_predict
            if hasattr(sk_cls, 'predict_proba'):
                __output_keys.add('pred_proba')
                __input_keys.add('X_test')

                def __do_predict_proba(self, **kwargs):
                    self.__fit(**kwargs)
                    (X_test, X_test_dtype) = self.__uo_to_np(kwargs['X_test'])
                    result = self.__sk_instance.predict_proba(X_test)
                    return self.__np_to_uo(result)
                __funcs_to_run['pred_proba'] = __do_predict_proba
            if hasattr(sk_cls, 'predict_log_proba'):
                __output_keys.add('pred_log_proba')
                __input_keys.add('X_test')

                def __do_predict_log_proba(self, **kwargs):
                    self.__fit(**kwargs)
                    (X_test, X_test_dtype) = self.__uo_to_np(kwargs['X_test'])
                    result = self.__sk_instance.predict_log_proba(X_test)
                    return self.__np_to_uo(result)
                __funcs_to_run['pred_log_proba'] = __do_predict_log_proba

        def run(self, outputs_requested, **kwargs):
            try:
                self.__params = kwargs['params_in'].to_dict()
            except KeyError:
                pass
            # If the user was inconsiderate enough to ask for probabilities
            #    without setting the probability param to True, we do it for
            #    them.
            if (('pred_proba' in outputs_requested or
                 'pred_log_proba' in outputs_requested) and
                 'probability' in inspect.getargspec(
                     self.__sk_cls.__init__).args):
                self.__params['probability'] = True
            self.__sk_instance = self.__sk_cls(**self.__params)
            self.__fitted = False
            return {output_key:
                    self.__funcs_to_run[output_key](self, **kwargs)
                    for output_key in outputs_requested}

        @property
        def input_keys(self):
            return list(self.__input_keys)

        @property
        def output_keys(self):
            return list(self.__output_keys)

        def get_sklearn_class(self):
            return self.__sk_cls

        def get_params(self):
            return self.__params

    return WrappedEstimator


def unpickle_metric(fun, args, kwargs):
    """ method used by pickle to unpickle wrapped metrics. Do not call
    directly """
    cls = __wrap_metric(fun)
    return cls(*args, **kwargs)

# metrics are whitelisted and we specify their return arguments manually
supported_metrics = {'roc_curve': ('fpr', 'tpr', 'thresholds'),
                     'precision_recall_curve': ('precision', 'recall', 
                                                'thresholds')}


def __wrap_metric(fun):
    func_name = fun.func_name
    if func_name not in supported_metrics:
        raise WrapSKLearnException('Not a supported metric')

    class WrappedMetric(RunnableStage):
        __argspec = inspect.getargspec(fun)
        __input_keys = __argspec.args[:-len(__argspec.defaults)]
        __output_keys = supported_metrics[func_name]

        def __fun(self, *args, **kwargs):
            return fun(*args, **kwargs)

        def __uo_to_np(self, uo):
            A_sa = uo.to_np()
            A, dtype = np_sa_to_nd(A_sa)
            return np.ravel(A)

        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __reduce__(self):
            return (unpickle_metric, (fun, self.__args, self.__kwargs))

        def run(self, outputs_requested, **kwargs):
            input_args = [self.__uo_to_np(kwargs[key]) for key in
                          self.__input_keys]
            np_out = self.__fun(
                *(input_args + list(self.__args)), **self.__kwargs)
            if len(self.__output_keys) <= 0:
                np_out = []
            elif len(self.__output_keys) == 1:
                np_out = [np_out]
            out = {key: UObject(UObjectPhase.Write) for key in
                   self.__output_keys}
            [out[key].from_np(np_nd_to_sa(np_out[i])) for i, key in
                enumerate(self.__output_keys)]
            return out

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys
    return WrappedMetric


def wrap(target):
    """returns a Stage class that wraps an sklearn object.

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | sklearn.metrics function | str
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass or a function in sklearn.metrics or the
        qualified backage name of a function in sklearn.metrics.

    Examples
    --------
    >>> from sklearn.preprocessing import Imputer
    >>> WrappedImputer = wrap(Imputer)
    >>> impute_stage = WrappedImputer()

    or

    >>> WrappedImputer = wrap('sklearn.preprocessing.Imputer')
    >>> impute_stage = WrappedImputer()

    >>> from sklearn.metrics import roc_curve
    >>> WrappedRoc = wrap(roc_curve)
    >>> roc_stage = WrappedRoc()

    >>> WrappedRoc = wrap('sklearn.metrics.roc_curve')
    >>> roc_stage = WrappedRoc()

    """
    if isinstance(target, str):
        split = target.split('.')
        object_name = split[-1]
        module_name = '.'.join(split[:-1])
        skl_module = importlib.import_module(module_name)
        skl_object = skl_module.__dict__[object_name]
    else:
        skl_object = target
    if isinstance(skl_object, FunctionType):  # Assuming this is a metric
        return __wrap_metric(skl_object)
    if issubclass(skl_object, sklearn.base.BaseEstimator):
        return __wrap_estimator(skl_object)
    raise TypeError(
        ('wrap takes a sklearn.base.BaseEstimator class '
         'or a function or a package name of one of the above objects'))


def wrap_instance(target, *args, **kwargs):
    """returns an instance of a Stage class that wraps an sklearn object.

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | sklearn.metrics function | str
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass or a function in sklearn.metrics or the
        qualified backage name of a function in sklearn.metrics.
    args:
        positional arguments to pass to constructor.
    kwargs:
        keyword arguments to pass to constructor

    Examples
    --------
    >>> from sklearn.preprocessing import Imputer
    >>> impute_stage = wrap_instance(Imputer, missing_values=0)

    or

    >>> impute_stage = wrap_instance('sklearn.preprocessing.Imputer',
            strategy='median')

    >>> roc_stage = wrap_instance('sklearn.metrics.roc_curve')

    >>> from sklearn.metrics import roc_curve
    >>> roc_stage = wrap_instance(roc_curve)

    """
    cls = wrap(target)
    return cls(*args, **kwargs)
