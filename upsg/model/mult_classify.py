import itertools as it
import numpy as np

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa
from .cross_validation import CrossValidationScore
from ..fetch.np import NumpyRead


class MultiClassify(MetaStage):

    # TODO declare which metrics
    """Runs data through a number of classifiers and a number of parameters
    per classifier. A search is performed for each classifier to find the 
    best of the given set of parameters. Finally, a report is constructed 
    giving some metrics for each classifier.

    Input Keys
    ----------
    X_train
    y_train
    X_test
    y_test

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
            return {'{}_out'.format(key): kwargs[key] for key in kwargs}


    def __init__(self, score_key, clf_and_params_dict=None, cv=2):
        """

        Parameters
        ----------
        clf_and_params_dict: dict of (
                (upsg.stage.Stage class or 
                 sklearn.base.BaseEstimator class or 
                 str): 
                (dict of str: list)
            )

            A dictionary signifying the classifiers and parameters to run.
            
            The keys of this dictionary should be either:
                1) classes that are subclasses of upsg.stage.Stage, or,
                2) classes that are sklearn classifiers, or
                3) strings that are the fully qualified package names 
                    sklearn classifiers.
             
             The values in the dictionary should be a set of parameters to try
             as in the params_dict argument of 
             upsg.model.grid_search.GridSearch

             If not provided, a default set of classifiers and parameters
             will be used. 

        score_key: str
            key output from clf_stage that should be used for scoring. 
                The table that the key stores should be of size 1x1

        cv : int (default 2)
            Number of cross-validation folds used to test a configuration.

        """

        raise NotImplementedError() 
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
