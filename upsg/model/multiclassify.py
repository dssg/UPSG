import itertools as it
import numpy as np
import json

from sklearn.base import BaseEstimator

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa, get_resource_path, utf_to_ascii
from ..wrap.wrap_sklearn import wrap, wrap_and_make_instance
from ..transform.split import SplitY
from ..transform.identity import Identity
from ..export.plot import Plot
from .grid_search import GridSearch
from .multimetric import Multimetric, VisualMetricSpec, NumericMetricSpec



class Multiclassify(MetaStage):

    # TODO declare which metrics
    """Runs data through a number of classifiers and a number of parameters
    per classifier. A search is performed for each classifier to find the 
    best of the given set of parameters. Finally, a report is constructed 
    giving some metrics for each classifier.

    **Input Keys**

    X_train

    y_train

    X_test

    y_test

    **Output Keys**

    report_file

    Parameters
    ----------
    clf_and_params_dict : dict of (
            (upsg.stage.Stage class or 
            sklearn.base.BaseEstimator class or 
            str) : 
            (dict of str : list)
        )

        A dictionary signifying the classifiers and parameters to run.
        
        The keys of this dictionary should be either:
            1. classes that are subclasses of upsg.stage.Stage, or,
            2. classes that are sklearn classifiers, or
            3. strings that are the fully qualified package names 
                sklearn classifiers.
         
         The values in the dictionary should be a set of parameters to try
         as in the params_dict argument of 
         upsg.model.grid_search.GridSearch

         If not provided, a default set of classifiers and parameters
         will be used. 

    score_key : str
        Key output from clf_stage that should be used for scoring. 
            The table that the key stores should be of size 1x1

    report_file_name : str
        Base name of file in which to write the report.

    cv : int (default 2)
        Number of cross-validation folds used to test a configuration.
    
    metrics : list of (upsg.model.multimetric.VisualMetricSpec or 
                       upsg.model.multimetric.NumericMetricSpec or
                       None)              
        Metrics to report for each classifier. If None, reports a 
        precision-recall, an ROC, and auc for the ROC

    """

    class __ReduceStage(RunnableStage):

        def __init__(self, classifiers, file_name):
            self.__file_name = file_name
            self.__classifiers = classifiers
            self.__width = len(classifiers)
            self.__report_keys = map('report_in{}'.format, 
                                     xrange(self.__width))
            self.__input_keys = self.__report_keys #+ self.__params_keys
            self.__output_keys = ['report_file']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            # TODO print reports in some nicer format
            with open(self.__file_name, 'w') as fout:
                fout.write('<!DOCTYPE html><html><body>')
                for i, classifier in enumerate(self.__classifiers):
                    with open(
                        kwargs['report_in{}'.format(i)].to_external_file()
                        ) as fin:
                        fout.write(fin.read())
                fout.write('</body></html>')
            uo_report_file = UObject(UObjectPhase.Write)
            uo_report_file.from_external_file(self.__file_name)
            return {'report_file': uo_report_file}


    def __init__(
            self, 
            score_key, 
            report_file_name, 
            clf_and_params_dict=None, 
            cv=2,
            metrics=None):

        """


        """
        if metrics is None:
            metrics = (VisualMetricSpec(
                           'sklearn.metrics.precision_recall_curve', # metric
                           'recall', # output key corresponding to x-axis
                           'precision', # output key corresponding to y-axis
                           'Precision/Recall Curve', # graph title
                           'Recall', # x-label
                           'Precision'), # y-label
                       VisualMetricSpec(
                           'sklearn.metrics.roc_curve',
                           None,
                           ('tpr', 'fpr'),
                           'ROC Curve',
                           'Results tagged positive',
                           'Rate',
                           ('FPR', 'TPR')),
                       NumericMetricSpec(
                           'sklearn.metrics.roc_auc_score',
                           'auc',
                           'ROC AUC Score'))

        if clf_and_params_dict is None:
            with open(
                get_resource_path(
                    'default_multi_classify.json')) as f_default_dict:
                clf_and_params_dict = json.load(f_default_dict)    
                
        classifiers = clf_and_params_dict.keys()
        p = Pipeline()
        self.__pipeline = p
        node_map = p.add(Identity(('X_train', 'y_train', 'X_test', 'y_test')))
        node_reduce = p.add(self.__ReduceStage(classifiers, report_file_name))

        for i, clf in enumerate(clf_and_params_dict):
            if (isinstance(clf, basestring) or
                issubclass(clf, BaseEstimator)): # an sklearn object
                clf_stage = wrap(clf)
            else: # presumably, a upsg.stage.Stage already
                #TODO type safety
                clf_stage = clf
            params = clf_and_params_dict[clf]
            node_grid_search = p.add(
                    GridSearch(
                        clf_stage, 
                        'score', 
                        params,
                        cv))
            node_map['X_train_out'] > node_grid_search['X_train']
            node_map['y_train_out'] > node_grid_search['y_train']
            node_map['X_test_out'] > node_grid_search['X_test']
            node_map['y_test_out'] > node_grid_search['y_test']

            node_proba_cat_1 = p.add(SplitY(-1))
            (node_grid_search['pred_proba'] > 
             node_proba_cat_1['input'])

            node_metric = p.add(Multimetric(metrics, str(clf)))
            node_proba_cat_1['y'] > node_metric['pred_proba']
            node_map['y_test_out'] > node_metric['y_true']
            node_grid_search['params_out'] > node_metric['params']
            if 'feature_importances' in node_grid_search.output_keys:
                (node_grid_search['feature_importances'] > 
                 node_metric['feature_importances'])

            (node_metric['report_file'] > 
             node_reduce['report_in{}'.format(i)])

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
