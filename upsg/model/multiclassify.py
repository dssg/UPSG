import itertools as it
import numpy as np
import json

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa, get_resource_path
from ..wrap.wrap_sklearn import wrap, wrap_and_make_instance
from ..transform.split import SplitColumn
from ..export.plot import Plot
from .grid_search import GridSearch


class Multiclassify(MetaStage):

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

    Output Keys
    -----------
    report_file

    """

    class __MapStage(RunnableStage):

        """Translates metastage input keys to input stage required by the
        stage

        """

        def __init__(self):
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

    class __ReduceStage(RunnableStage):

        def __init__(self, classifiers, file_name):
            self.__file_name = file_name
            self.__classifiers = classifiers
            self.__width = len(classifiers)
            self.__params_keys = map('params_in{}'.format, 
                                     xrange(self.__width))
            self.__report_keys = map('report_in{}'.format, 
                                     xrange(self.__width))
            self.__input_keys = self.__report_keys + self.__params_keys
            self.__output_keys = ['report_file']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def __print_classifier_report(self, fout, classifier, uo_params, 
                                      uo_report):
            fout.write(
                '<h3>{}</h3><p>Best params: {}<p><img src="{}">'.format(
                    classifier,
                    uo_params.to_dict(),
                    uo_report.to_external_file()))

        def run(self, outputs_requested, **kwargs):
            # TODO print reports in some nicer format
            with open(self.__file_name, 'w') as fout:
                fout.write('<!DOCTYPE html><html><body>')
                for i, classifier in enumerate(self.__classifiers):
                    self.__print_classifier_report(
                        fout,
                        classifier,
                        kwargs['params_in{}'.format(i)],
                        kwargs['report_in{}'.format(i)])
                fout.write('</body></html>')
            uo_report_file = UObject(UObjectPhase.Write)
            uo_report_file.from_external_file(self.__file_name)
            return {'report_file': uo_report_file}


    def __init__(self, score_key, report_file_name, clf_and_params_dict=None, cv=2):
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
            Key output from clf_stage that should be used for scoring. 
                The table that the key stores should be of size 1x1

        report_file_name: str
            Base name of file in which to write the report.

        cv : int (default 2)
            Number of cross-validation folds used to test a configuration.

        """

        if clf_and_params_dict is None:
            with open(
                get_resource_path(
                    'default_multi_classify.json')) as f_default_dict:
                clf_and_params_dict = json.load(f_default_dict)    

        classifiers = clf_and_params_dict.keys()
        p = Pipeline()
        self.__pipeline = p
        node_map = p.add(self.__MapStage())
        node_reduce = p.add(self.__ReduceStage(classifiers, report_file_name))

        for i, clf in enumerate(clf_and_params_dict):
            if (isinstance(clf, str) or 
                issubclass(sklearn.base.BaseEstimator)): # an sklearn object
                clf_stage = wrap(clf)
            else: # presumably, a upsg.stage.Stage already
                clf_stage = clf
            node_grid_search = p.add(
                    GridSearch(
                        clf_stage, 
                        'score', 
                        clf_and_params_dict[clf],
                        cv))
            node_map['X_train_out'] > node_grid_search['X_train']
            node_map['y_train_out'] > node_grid_search['y_train']
            node_map['X_test_out'] > node_grid_search['X_test']
            node_map['y_test_out'] > node_grid_search['y_test']

            node_proba_cat_1 = p.add(SplitColumn(-1))
            node_grid_search['pred_proba'] > node_proba_cat_1['in']

            node_calc_precision_recall = p.add(
                wrap_and_make_instance(
                    'sklearn.metrics.precision_recall_curve'))
                
            node_proba_cat_1['y'] > node_calc_precision_recall['probas_pred']
            node_map['y_test_out'] > node_calc_precision_recall['y_true']

            node_plot_calc_precision_recall = p.add(
                Plot(
                    'calc_precision_recall{}.png'.format(i),
                    xlabel='Recall',
                    ylabel='Precision'))
            (node_calc_precision_recall['recall'] > 
             node_plot_calc_precision_recall['x'])
            (node_calc_precision_recall['precision'] > 
             node_plot_calc_precision_recall['y'])

            (node_plot_calc_precision_recall['plot_file'] > 
             node_reduce['report_in{}'.format(i)])
            (node_grid_search['params_out'] > 
             node_reduce['params_in{}'.format(i)])

        self.__in_node = node_map
        self.__out_node = node_reduce

    @property
    def input_keys(self):
        return self.__in_node.input_keys

    @property
    def output_keys(self):
        return self.__out_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
