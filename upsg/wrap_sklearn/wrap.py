import importlib

from ..stage import Stage
import sklearn.base
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
                X = kwargs['X'].to_np()
                print X
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
                X_new = self.__sk_instance.fit_transform(X, y, **fit_params)
                print X_new
                uo_out = UObject(UObjectPhase.Write)
                uo_out.from_np(X_new)
                return {'X_new' : uo_out}                
   
        else:
            #TODO we should be able to accomidate different sorts of classes
            raise NotImplementedError('For now, we can only wrap scikit transformers')
        

        @property 
        def input_keys():
            return __input_keys

        @property
        def output_keys():
            return __output_keys

            

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
