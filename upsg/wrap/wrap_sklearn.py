import importlib

import sklearn.base
import numpy as np

from ..stage import Stage
from ..uobject import UObject, UObjectPhase
from ..utils import np_nd_to_sa, np_sa_to_nd

def unpickle_constructor(sk_instance):
    cls = __wrap_class(type(sk_instance))
    return cls(sk_instance) 

def __wrap_class(sk_cls):
    """Wraps a scikit BaseEstimator class inside a UPSG Stage and returns it
    """
    class Wrapped(Stage):
        __sk_cls = sk_cls

        def __init__(self, sk_instance = None, **kwargs):
            if sk_instance is None:
                self.__sk_instance = self.__sk_cls(**kwargs)
            else:
                self.__sk_instance = sk_instance
            self.__cached_uos = {}

        def __reduce__(self):
            return (unpickle_constructor, (self.__sk_instance,))

        def __uo_to_np(self, uo):
            try:
                (A, dtype) = self.__cached_uos[uo]
            except KeyError:
                A_sa = uo.to_np()
                A, dtype = np_sa_to_nd(A_sa)
                self.__cached_uos[uo] = (A, dtype)
            return (A, dtype)  

        def __np_to_uo(self, A, dtype = None):
            A_sa = np_nd_to_sa(A, dtype)
            uo_out = UObject(UObjectPhase.Write)
            uo_out.from_np(A_sa)
            return uo_out

        __input_keys = set()
        __output_keys = set()
        __funcs_to_run = {}
        # It would be nicer to use class hierarchy than hasattr, but sklearn
        # doesn't put everything in interfaces
        #if issubclass(sk_cls, sklearn.base.TransformerMixin):
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
                #TODO
                #try:
                #    fit_params = kwargs['fit_params'].to_dict()
                #except KeyError:
                #    fit_params = {}
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
                self.__sk_instance.fit(X_train, np.ravel(y_train), 
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
                    score = self.__sk_instance.score(X_test, np.ravel(y_test), 
                        sample_weight)
                    return self.__np_to_uo(np.array([[score]]), [('score', 
                        type(score))])
                __funcs_to_run['score'] = __do_score 
            if hasattr(sk_cls, 'predict'):
                __output_keys.add('y_pred')
                __input_keys.add('X_test') 
                def __do_predict(self, **kwargs):
                    self.__fit(**kwargs)
                    (X_test, X_test_dtype) = self.__uo_to_np(kwargs['X_test'])
                    y_pred = self.__sk_instance.predict(X_test)
                    return self.__np_to_uo(y_pred.reshape(len(y_pred), -1))
                __funcs_to_run['y_pred'] = __do_predict
            
        def run(self, outputs_requested, **kwargs):
            self.__fitted = False
            return {output_key : 
                self.__funcs_to_run[output_key](self, **kwargs) 
                for output_key in outputs_requested}

        @property 
        def input_keys(self):
            return list(self.__input_keys)

        @property
        def output_keys(self):
            return list(self.__output_keys)

        def get_sklearn_instance(self):
            return self.__sk_instance           

        def get_params(self):
            return self.__sk_instance.get_params()

    return Wrapped

def wrap(target):
    """returns a Stage class that wraps an sklearn object.

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | string
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass.

    Examples
    --------
    >>> from sklearn.preprocessing import Imputer
    >>> WrappedImputer = wrap(Imputer) 
    >>> impute_stage = WrappedImputer()
    
    or

    >>> WrappedImputer = wrap('sklearn.preprocessing.Imputer') 
    >>> impute_stage = WrappedImputer()
    
    """
    if isinstance(target, str):
        split = target.split('.')
        cls_name = split[-1]
        module_name = '.'.join(split[:-1])
        skl_module = importlib.import_module(module_name)
        skl_class = skl_module.__dict__[cls_name]
    else:
        skl_class = target 
    if not issubclass(skl_class, sklearn.base.BaseEstimator):
        raise TypeError(('wrap takes a sklearn.base.BaseEstimator class ' 
            'or a string'))
    return __wrap_class(skl_class)

def wrap_instance(target, *args, **kwargs):
    """returns an instance of a Stage class that wraps an sklearn object.

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | string
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass.
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
    
    """
    cls = wrap(target)
    return cls(*args, **kwargs)
