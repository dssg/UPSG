import numpy as np
from types import FunctionType
import inspect
from operator import itemgetter
import itertools as it

import sklearn.base
import sklearn.cross_validation

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_nd_to_sa, np_sa_to_nd, import_object_by_name


class WrapSKLearnException(Exception):
    pass


def __wrap_partition_iterator(sk_cls):
    """Wraps a subclass of sklearn.cross_validation._PartitionIterator"""
    class WrappedPartitionIterator(RunnableStage):
        __sk_cls = sk_cls
        expected_kwargs = inspect.getargspec(__sk_cls.__init__).args

        if 'n_folds' not in expected_kwargs:
            raise WrapSKLearnException(('LeaveOut Partition iterators are not '
                                        'supported yet.'))

        def __init__(self, n_arrays=1, n_folds=2, **kwargs):
            self.__n_arrays = n_arrays
            self.__n_folds = n_folds
            self.__kwargs = kwargs
            self.__expected_kwargs = self.expected_kwargs
            self.__in_array_keys = ['input{}'.format(array) for array in 
                                 xrange(n_arrays)]
            self.__input_keys = list(self.__in_array_keys)
            if 'y' in self.__expected_kwargs:
                self.__input_keys.append('y')
            self.__output_keys = list(it.chain.from_iterable(
                    (('train{}_{}'.format(array, fold), 
                      'test{}_{}'.format(array, fold))
                     for array, fold in it.product(
                         xrange(n_arrays), xrange(n_folds)))))
        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            in_arrays = [kwargs[key].to_np() for key in self.__in_array_keys]
            if len(in_arrays) < 1:
                return {}
            pi_kwargs = self.__kwargs
            # TODO introspect sk_cls input args to figure out what it needs to take
            if 'n' in self.__expected_kwargs:
                pi_kwargs['n'] = in_arrays[0].shape[0]
            if 'n_folds' in self.__expected_kwargs:
                pi_kwargs['n_folds'] = self.__n_folds
            if 'y' in self.__expected_kwargs:
                pi_kwargs['y'] = np_sa_to_nd(kwargs['y'].to_np())[0]
            pi = self.__sk_cls(**self.__kwargs)
            results = {key: UObject(UObjectPhase.Write) for key
                       in self.__output_keys}
            for fold_index, (train_inds, test_inds) in enumerate(pi):
                for array_index, in_key in enumerate(self.__in_array_keys):
                    key_number = int(in_key.replace('input', ''))
                    results['train{}_{}'.format(key_number, fold_index)].from_np(
                        in_arrays[array_index][train_inds])
                    results['test{}_{}'.format(key_number, fold_index)].from_np(
                        in_arrays[array_index][test_inds])
            return results

    return WrappedPartitionIterator

def unpickle_estimator(sk_cls, params):
    """ 
    
    method used by pickle to unpickle wrapped estimators. Do not call
    directly 
    
    """
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

        def __repr__(self):
            # Not really how we init these, but it gets the point accross
           return 'WrappedEstimator({}, {})'.format(
                    self.__sk_cls, self.__params)

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
                X_new_nd = self.__sk_instance.fit_transform(X_train, y_train,
                                                            **fit_params)
                if X_new_nd.shape[1] < len(X_train_dtype): 
                    # We have lost some columns. Presumably our Estimator did
                    # feature selection, so we can ask for its support
                    try:
                        support = self.__sk_instance.get_support()
                    except AttributeError: 
                        # doesn't have a get support method. try for
                        # coeficients:
                        # http://stackoverflow.com/questions/25007640/sklearn-get-feature-names-after-l1-based-feature-selection
                        support = np.any(self.__sk_instance.coef_ != 0, 0)  
                    X_train_dtype = np.dtype(
                            [dt for dt, support in 
                             zip(
                                 X_train_dtype.descr, 
                                 support) if
                             support])

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
                if 'sample_weight' in inspect.getargspec(
                    self.__sk_instance.fit).args:
                    self.__sk_instance.fit(X_train, y_train,
                                       sample_weight)
                else:
                    self.__sk_instance.fit(X_train, y_train)
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

            if hasattr(sk_cls, 'feature_importances_'):
                __output_keys.add('feature_importances')

                def __do_feature_importances(self, **kwargs):
                    self.__fit(**kwargs)
                    clf = self.__sk_instance
                    (X_train, X_train_dtype) = self.__uo_to_np(kwargs['X_train'])
                    col_names = X_train_dtype.names
                    feat_importances = [ (col_names[x], 
                                          clf.feature_importances_[x]) for
                        x in clf.feature_importances_.argsort()[::-1]]
                    col_name_chars = len(max(
                        feat_importances, 
                        key=lambda feat: len(feat[0]))[0])
                    np_importances = np.array(
                            feat_importances,
                            dtype=[('col_name', 'S{}'.format(col_name_chars)), 
                                   ('rank', float)])
                    return self.__np_to_uo(np_importances) 
                __funcs_to_run['feature_importances'] = __do_feature_importances
                    

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
    """ 
    
    method used by pickle to unpickle wrapped metrics. Do not call
    directly 
    
    """
    cls = __wrap_metric(fun)
    return cls(*args, **kwargs)

# metrics are whitelisted and we specify their return arguments manually
supported_metrics = {'roc_curve': ('fpr', 'tpr', 'thresholds'),
                     'precision_recall_curve': ('precision', 'recall', 
                                                'thresholds'),
                     'roc_auc_score': ('auc',)}


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
                np_out = ()
            elif len(self.__output_keys) == 1:
                np_out = (np_out,)
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
    """Returns a Stage class that wraps an sklearn object.

    **Input Keys and Output Keys**

    Input keys and output keys depend on the class being wrapped. If the 
    sklearn object being wrapped is an estimator:

    1. The Stage will always have a "params_in" input key, which is equivalent
       to the arguments of the 
       `BaseEstimator.set_params <http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_params>`_ 
       method. These parameters can be generated by the output key that all 
       wrapping Stages have: "params_out". "params_out" is equivalent to the 
       results of the:
       `BaseEstimator.get_params <http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params>`_
       method.
    2. If the estimator implements fit_transform, then the Stage will provide the
       input keys: "X_train", "y_train", and "fit_params" which correspond to 
       the X, y, and fit_params arguments of the
       `TransformerMixin.fit_transform <http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin.fit_transform>`_
       method. The Stage will also provide the output key "X_new", which 
       corresponds to the result of fit_transform.
    3. If the estimator implements fit, then the Stage provides the input keys 
       "X_train", "Y_train",
       and "sample_weight", which correspond to the X, y, and sample_weight 
       arguments of fit.
    4. If the estimator implements score, then the Stage will provide the 
       input keys "X_test" and "y_test" which corresponds to the X and y 
       arguments of score. The Stage also provides the output key "score".
    5. If the estimator implements predict, then the Stage will provide the 
       input key "X_test" which corresponds to the X argument of predict. 
       The Stage also provides the output key "y_pred", which is the
       predicted y values.
    6. If the estimator implements score, then the Stage will provide the 
       input keys "X_test" and "y_test" which corresponds to the X and y 
       arguments of score. The Stage also provides the output key "Score".
    7. If the estimator implements predict, then the Stage will provide the 
       input key "X_test" which corresponds to the X arguments of predict. 
       The Stage also provides the output key "y_pred", which is the
       predicted y values.
    8. If the estimator implements predict_proba and/or predict_log_proba,
       the Stage will provide the input key "X_test", which corresponds to the
       X argument of predict_proba and predict_log_proba. The stage will 
       provide the output keys "pred_proba" and "pred_log_proba" respectively.

    If the object being wrapped is a metric, the input keys will have the
    same names as the arguments to the metric. The output keys are
    currently assigned arbitrarily. At present, output keys for supported
    metric are:

    roc_curve
        "fpr", "tpr", "thresholds"
    precision_recall_curve
        "precision", "recall", "thresholds"
    roc_auc_score
        "auc"

    If the object being wrapped is a subclass of 
    sklearn.cross_validation._PartitionIterator, (i.e. sklearn.cross_validation.KFold) 
    the returned class must be initialized with an n_arrays argument and an 
    n_folds argument. There are n_arrays input arrays called:

    'input0', 'input1', 'input2', ...

    If the _PartitionIterator takes a 'y' argument (e.g. StratifiedKFold) then
    there is additionally a 'y' input key

    Depending on the number of folds requested (n_folds) output keys will be:

        'train0_0', 'test0_0', 'train0_1', 'test0_1',... (corresponding to
        different folds of 'input0')

        'train1_0', 'test1_0', 'train1_1', 'test1_1',... (corresponding to
        different folds of 'input1')

        'train2_0', 'test2_0', 'train2_1', 'test2_1',... (corresponding to
        different folds of 'input2')

        etc. 

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | sklearn.metrics function | sklearn.cross_validation._PartitionIterator class | str
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass or a function in sklearn.metrics or the
        qualified backage name of a function in sklearn.metrics or the 
        _PartitionIterator subclass or the fully qualified package name of the
        _PartitionIterator.

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

    or

    >>> WrappedKFold = wrap('sklearn.cross_validation.KFold')
    >>> kfold_stage = WrappedKFold(n_arrays=2, n_folds=3)

    """
    #TODO add partitioniterator to doc
    if isinstance(target, basestring):
        skl_object = import_object_by_name(target)
    else:
        skl_object = target
    if isinstance(skl_object, FunctionType):  # Assuming this is a metric
        return __wrap_metric(skl_object)
    if issubclass(skl_object, sklearn.base.BaseEstimator):
        return __wrap_estimator(skl_object)
    if issubclass(skl_object, sklearn.cross_validation._PartitionIterator):
        return __wrap_partition_iterator(skl_object)
    raise TypeError(
        ('wrap takes a sklearn.base.BaseEstimator class '
         'or a function or a package name of one of the above objects'))


def wrap_and_make_instance(target, **kwargs):
    """Returns an instance of a Stage class that wraps an sklearn object.

    Parameters
    ----------
    target: sklearn.base.BaseEstimator class | sklearn.metrics function | sklearn.cross_validation._PartitionIterator class | str
        Either a BaseEstimator subclass or the fully qualified package name
        of a BaseEstimater subclass or a function in sklearn.metrics or the
        qualified backage name of a function in sklearn.metrics or the 
        _PartitionIterator subclass or the fully qualified package name of the
        _PartitionIterator.
    args:
        positional arguments to pass to constructor.
    kwargs:
        keyword arguments to pass to constructor

    Examples
    --------
    >>> from sklearn.preprocessing import Imputer
    >>> impute_stage = wrap_and_make_instance(Imputer, missing_values=0)

    or

    >>> impute_stage = wrap_and_make_instance('sklearn.preprocessing.Imputer',
            strategy='median')

    >>> roc_stage = wrap_and_make_instance('sklearn.metrics.roc_curve')

    >>> from sklearn.metrics import roc_curve
    >>> roc_stage = wrap_and_make_instance(roc_curve)

    or 

    >>> kfold_stage = wrap_and_make_instance('sklearn.cross_validation.KFold',
            n_arrays=2, n_folds=3)

    """
    cls = wrap(target)
    return cls(**kwargs)
