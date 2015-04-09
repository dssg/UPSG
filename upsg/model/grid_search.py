import itertools as it
import numpy as np

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa
from .cross_validation import CrossValidationScore
from ..fetch.np import NumpyRead


class GridSearch(MetaStage):

    """Searches over a grid of parameters for a given classifier and finds the
    set of parameters that score the best.

    Then, the classifier with the best parameters can be used to make a
    predicition

    Input Keys
    ----------
    X_train
    y_train
    X_test
    y_test

    Output Keys
    -----------
    y_pred : predicted y data corresponding to X_test
    params : the best parameters found

    """

    class __MapStage(RunnableStage):

        """Translates metastage input keys to input stage required by the
        stage

        """
        # Just passes the values on for now. It might need to modify them later

        def __init__(self, n_children):
            self.__n_children = n_children
            self.__input_keys = ['X_train', 'y_train', 'X_test', 'y_test']
            self.__output_keys = ['{}_out'.format(key) for key
                                  in self.__input_keys]

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            #            ret_hier = ({out_key : kwargs[in_key] for out_key
            #                in self.__output_keys_hier[in_key]} for in_key
            #                in self.__input_keys)
            #            ret = {}
            #            map(ret.update, ret_hier)
            #            return ret
            return {'{}_out'.format(key): kwargs[key] for key in kwargs}

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

    def __init__(self, clf_stage, score_key, params_dict, cv=2):
        """

        Parameters
        ----------
        clf_stage : Stage class
            class of a Stage for which parameters will be tested
        score_key : str
            key output from clf_stage that should be used for scoring. 
                The table that the key stores should be of size 1x1
        params_dict : dict of (string : list)
            A dictionary where the keys are parameters and their values are a
            list of values to try for that paramter. For example, if given
            {'param1' : [1, 10], 'param2' : ['a', 'b']}, GridSearch will
            search for the highest-scoring among clf_stage(param1 = 1,
            param2 = 'a'), clf_stage(param1 = 1, param2 = 'b'),
            clf_stage(param1 = 10, param2 = 'b'), clf_stage(param1 = 10,
            param2 = 'b')
        cv : int (default 2)
            Number of cross-validation folds used to test a configuration.

        """

        self.__clf_stage = clf_stage
        # produces dictionaries of the cartesian product of our parameters.
        # from
        # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        self.__params_prod = [dict(it.izip(params_dict, x))
                              for x in it.product(*params_dict.itervalues())]
        width = len(self.__params_prod)
        p = Pipeline()
        self.__pipeline = p
        node_map = p.add(self.__MapStage(width))
        node_reduce = p.add(self.__ReduceStage(width))
        node_final = p.add(clf_stage())

        for i, params in enumerate(self.__params_prod):
            node_cv_score = p.add(
                    CrossValidationScore(clf_stage, score_key, params, cv))
            node_map['X_train_out'] > node_cv_score['X_train']
            node_map['y_train_out'] > node_cv_score['y_train']

            node_params = p.add(NumpyRead(dict_to_np_sa(params)))

            node_cv_score['score'] > node_reduce['score_in{}'.format(i)]
            node_params['out'] > node_reduce['params_in{}'.format(i)]

        [node_map['{}_out'.format(key)] > node_final[key] for key in
            ['X_train', 'X_test', 'y_train', 'y_test']]
        node_reduce['params_out'] > node_final['params_in']
        self.__in_node = node_map
        self.__out_node = node_final

    @property
    def input_keys(self):
        return ['X_train', 'y_train', 'X_test', 'y_test']

    @property
    def output_keys(self):
        return ['y_pred', 'params']

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
