import itertools as it
import numpy as np

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa, utf_to_ascii
from .cross_validation import CrossValidationScore
from ..fetch.np import NumpyRead
from ..transform.identity import Identity


class GridSearch(MetaStage):

    """Searches over a grid of parameters for a given classifier and finds the
    set of parameters that score the best.

    Then, the classifier with the best parameters can be used to make a
    predicition

    **Input Keys**

    X_train

    y_train

    X_test

    y_test

    **Output Keys**

    y_pred 
        predicted y data corresponding to X_test for the best parameters
    pred_proba 
        class probabilities for the best parameters
    params 
        the best parameters found

    Parameters
    ----------
    clf_stage : Stage class
        class of a Stage for which parameters will be tested
    clf_stage_sweep_params : dict of (string : list)
        A dictionary where the keys are parameters and their values are a
        list of values to try for that paramter. For example, if given
        {'param1' : [1, 10], 'param2' : ['a', 'b']}, GridSearch will
        search for the highest-scoring among clf_stage(param1 = 1,
        param2 = 'a'), clf_stage(param1 = 1, param2 = 'b'),
        clf_stage(param1 = 10, param2 = 'b'), clf_stage(param1 = 10,
        param2 = 'b')
    score_key : str
        key output from clf_stage that should be used for scoring. 
            The table that the key stores should be of size 1x1
    cv_stage : Stage class or None
        class of stage to provide cross-validate partitioning.
        For example, 
        upsg.wrap.wrap_sklearn.wrap('sklearn.cross_validation.KFold')
        If None, then
        upsg.wrap.wrap_sklearn.wrap('sklearn.cross_validation.KFold')
        is used.
    cv_stage_kwargs: dict of str: ?
        Arguments corresponding to the keyword arguments of
        sklearn.cross_validation.KFold including n_folds

    """

    class __ReduceStage(RunnableStage):

        def __init__(self, n_parents):
            self.__n_parents = n_parents
            self.__score_keys = map('score_in{}'.format, range(n_parents))
            self.__params_keys = map('params_in{}'.format, range(n_parents))
            self.__input_keys = self.__score_keys + self.__params_keys
            self.__output_keys = ['params_out']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            # TODO return data in a format that tells you what the params were
            scores_array = np.array(
                [kwargs[key].to_np()[0][0] for key in self.__score_keys])
            best = kwargs[self.__params_keys[np.argsort(scores_array)[-1]]]
            return {'params_out': best}

    def __init__(self, clf_stage, clf_stage_sweep_params={}, score_key='score', 
                 cv_stage=None, cv_stage_kwargs={}):

        self.__clf_stage = clf_stage
        # produces dictionaries of the cartesian product of our parameters.
        # from
        # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        self.__params_prod = [dict(it.izip(clf_stage_sweep_params, x))
                              for x in it.product(
                                  *clf_stage_sweep_params.itervalues())]
        width = len(self.__params_prod)
        p = Pipeline()
        self.__pipeline = p
        node_map = p.add(Identity(('X_train', 'y_train', 'X_test', 'y_test')))
        node_reduce = p.add(self.__ReduceStage(width))
        node_final = p.add(clf_stage())

        for i, params in enumerate(self.__params_prod):
            node_cv_score = p.add(
                    CrossValidationScore(
                        clf_stage, 
                        {key: utf_to_ascii(params[key]) for key in params},
                        score_key, 
                        cv_stage, 
                        cv_stage_kwargs))
            node_map['X_train_out'] > node_cv_score['X_train']
            node_map['y_train_out'] > node_cv_score['y_train']

            node_params = p.add(NumpyRead(dict_to_np_sa(params)))

            node_cv_score['score'] > node_reduce['score_in{}'.format(i)]
            node_params['output'] > node_reduce['params_in{}'.format(i)]

        [node_map['{}_out'.format(key)] > node_final[key] for key in
            ['X_train', 'X_test', 'y_train', 'y_test']]
        node_reduce['params_out'] > node_final['params_in']
        self.__in_node = node_map
        self.__out_node = node_final

    @property
    def input_keys(self):
        return self.__in_node.input_keys

    @property
    def output_keys(self):
        return self.__out_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
