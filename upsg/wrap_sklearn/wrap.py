import importlib

import sklearn.base
import numpy as np

from ..stage import Stage
from ..uobject import UObject, UObjectPhase


def __wrap_class(sk_cls):
    """Wraps a scikit BaseEstimator class inside a UPSG Stage and returns it
    """
    class Wrapped(Stage):
        __sk_cls = sk_cls

        def __init__(self, **kwargs):
            self.__sk_instance = self.__sk_cls(**kwargs)

        if issubclass(sk_cls, sklearn.base.TransformerMixin):
            __input_keys = {'X': True, 'y' : False, 'fit_params' : False}
            __output_keys = ['X_new']
            def run(self, **kwargs):
                X_sa = kwargs['X'].to_np()
                X = X_sa.view(dtype=X_sa[0][0].dtype).reshape(
                    len(X_sa), -1)
                try:
                    y = kwargs['y'].to_np()
                except KeyError:
                    y = None
                #TODO
                #try:
                #    fit_params = kwargs['fit_params'].to_dict()
                #except KeyError:
                #    fit_params = {}
                fit_params = {}
                X_new_nd = self.__sk_instance.fit_transform(X, y, **fit_params)
                X_new = X_new_nd.view(dtype=X_sa.dtype).reshape(
                    len(X_sa))
                uo_out = UObject(UObjectPhase.Write)
                uo_out.from_np(X_new)
                return {'X_new' : uo_out}                
   
        else:
            #TODO we should be able to accomidate different sorts of classes
            raise NotImplementedError('For now, we can only wrap scikit transformers')
        

        @property 
        def input_keys(self):
            return __input_keys

        @property
        def output_keys(self):
            return __output_keys

        def get_sklearn_instance(self):
            return self.__sk_instance           

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
