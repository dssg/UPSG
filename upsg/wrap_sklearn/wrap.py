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
            self.__cached_uos = {}

        def __uo_to_np(self, uo):
            try:
                A = self.__cached_uos[uo]
            except KeyError:
                A_sa = uo.to_np()
                A = A_sa.view(dtype=A_sa[0][0].dtype).reshape(
                    len(A_sa), -1)
                self.__cached_uos[uo] = A
            return (A, A_sa.dtype)  

        def __np_to_uo(self, A, dtype):
            A_sa = A.view(dtype=dtype).reshape(
                A.shape[0])
            uo_out = UObject(UObjectPhase.Write)
            uo_out.from_np(A_sa)
            return uo_out

        __input_keys = {}
        __output_keys = set()
        __funcs_to_run = {}
        # It would be nicer to use class hierarchy than hasattr, but sklearn
        # doesn't put everything in interfaces
        #if issubclass(sk_cls, sklearn.base.TransformerMixin):
        if hasattr(sk_cls, 'fit_transform'):
            __input_keys['X'] = True 
            __input_keys['y'] = False 
            __input_keys['fit_params'] = False
            __output_keys.add('X_new')
            def do_fit_transform(self, **kwargs):
                (X, X_dtype) = self.__uo_to_np(kwargs['X'])
                try:
                    (y, y_dtype) = self.__uo_to_np(kwargs['y'])
                except KeyError:
                    y = None
                #TODO
                #try:
                #    fit_params = kwargs['fit_params'].to_dict()
                #except KeyError:
                #    fit_params = {}
                fit_params = {}
                X_new_nd = self.__sk_instance.fit_transform(X, y, **fit_params)
                return self.__np_to_uo(X_new_nd, X_dtype)
            __funcs_to_run['X_new'] = do_fit_transform
        if hasattr(sk_cls, 'score'):
            def do_score(self, **kwargs):
                pass
                #TODO sub
   
        def run(self, outputs_requested, **kwargs):
            return {output_key : 
                self.__funcs_to_run[output_key](self, **kwargs) 
                for output_key in outputs_requested}

        @property 
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return list(self.__output_keys)

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
