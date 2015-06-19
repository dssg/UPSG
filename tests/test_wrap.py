import numpy as np
from os import system
import unittest
from inspect import getargspec
import string
import random
import json

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

from upsg.wrap.wrap_sklearn import wrap, wrap_and_make_instance
from upsg.uobject import UObject, UObjectPhase
from upsg.pipeline import Pipeline
from upsg.fetch.csv import CSVRead
from upsg.fetch.np import NumpyRead
from upsg.export.csv import CSVWrite
from upsg.export.plot import Plot
from upsg.transform.split import SplitY, SplitTrainTest
from upsg.utils import np_nd_to_sa, np_sa_to_nd, get_resource_path
from upsg.utils import import_object_by_name

from utils import path_of_data, UPSGTestCase, csv_read


class TestWrap(UPSGTestCase):

    def test_from_module(self):
        WrappedImputer = wrap(Imputer)
        impute_stage = WrappedImputer()

    def test_from_string(self):
        WrappedImputer = wrap('sklearn.preprocessing.Imputer')
        impute_stage = WrappedImputer()

    def test_wrap_and_make_instance(self):
        impute_stage = wrap_and_make_instance('sklearn.preprocessing.Imputer',
                                     strategy='median')
        params = impute_stage.get_params()
        self.assertEqual(params['strategy'], 'median')

    def __process_in_data(self, in_data):
        if in_data is None:
            return (np.random.random((100,10)), 
                    np.random.randint(0, 2, 100))
        elif isinstance(in_data, str) and in_data.split('.')[-1] == 'csv':
            a = np_sa_to_nd(csv_read(path_of_data(in_data)))[0]
            return (a[:, :-1], a[:, -1])
        # assume in_data is a tuple (X, y)
        return (in_data[0], in_data[1])

    def __simple_pipeline(self, sk_cls, sk_method_name, upsg_out_key, 
                          init_kwargs={}, in_data=None):
        
        X_in, y_in = self.__process_in_data(in_data)

        ctrl_sk_inst = sk_cls(**init_kwargs)
        est_params = ctrl_sk_inst.get_params()
        try:
            random_state = est_params['random_state']
            if random_state is None:
                # This has to be fixed. Set a state and try again
                init_kwargs['random_state'] = 0
                ctrl_sk_inst = sk_cls(**init_kwargs)
        except KeyError:
            pass

        p = Pipeline()

        sk_stage = p.add(wrap_and_make_instance(
            sk_cls, 
            **init_kwargs))

        X_in_stage = p.add(NumpyRead(X_in))
        y_in_stage = p.add(NumpyRead(y_in))

        if sk_method_name == 'predict':
            train_test = p.add(SplitTrainTest(2, random_state=0))
            X_in_stage['output'] > train_test['input0']
            y_in_stage['output'] > train_test['input1']

            input_keys = sk_stage.get_stage().input_keys
            if 'X_train' in input_keys:
                train_test['train0'] > sk_stage['X_train']
            if 'X_test' in input_keys:
                train_test['test0'] > sk_stage['X_test']
            if 'y_train' in input_keys:
                train_test['train1'] > sk_stage['y_train']
        else:
            X_in_stage['output'] > sk_stage['X_train']
            y_in_stage['output'] > sk_stage['y_train']

        csv_out = p.add(CSVWrite(self._tmp_files.get('out.csv')))
        sk_stage[upsg_out_key] > csv_out['input']

        self.run_pipeline(p)

        if sk_method_name == 'predict':
            ctrl_X_train, ctrl_X_test, ctrl_y_train, ctrl_y_test = (
                train_test_split(X_in, y_in, random_state=0))
            ctrl_sk_inst.fit(ctrl_X_train, ctrl_y_train)
            control = ctrl_sk_inst.predict(ctrl_X_test)
        else:
            control = ctrl_sk_inst.fit_transform(X_in, y_in)

        result = self._tmp_files.csv_read('out.csv', as_nd=True)
        if result.ndim != control.ndim and result.ndim == 1:
            result = result.reshape(result.size, 1)

        self.assertTrue(result.shape == control.shape and 
                        np.allclose(result, control))

    def test_transform(self):
        kwargs = {'strategy': 'mean', 'missing_values': 'NaN'}
        self.__simple_pipeline(Imputer, 'transform', 'X_new', 
                               init_kwargs=kwargs, in_data='missing_vals.csv')

    def test_predict(self):

        with open(
            get_resource_path(
                'default_multi_classify.json')) as f_default_dict:
            clf_and_params_dict = json.load(f_default_dict)    
         
        for clf in clf_and_params_dict:
            self.__simple_pipeline(
                    import_object_by_name(clf), 
                    'predict', 
                    'y_pred')

    def test_factor_selection(self):
        # These are based on the documentation: 
        # http://scikit-learn.org/stable/modules/feature_selection.html
        # and
        # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#example-feature-selection-plot-rfe-digits-py
        vt_in =  (np.array([[0, 0, 1], 
                            [0, 1, 0], 
                            [1, 0, 0], 
                            [0, 1, 1], 
                            [0, 1, 0], 
                            [0, 1, 1]]),
                  np.zeros(6))
        iris = datasets.load_iris()
        in_svc = (iris.data, iris.target)
        trials = [(VarianceThreshold, {'threshold': (.8 * (1 - .8))}, 
                   vt_in),
                  (SelectKBest, {'score_func': chi2, 'k': 2}, None),
                  (RFE, {'estimator': SVC(kernel="linear", C=1), 
                             'n_features_to_select': 1,
                             'step': 1}, None),
                  (LinearSVC, {'C': 0.01, 'penalty': "l1", 'dual': False},
                   in_svc)]
        for clf, kwargs, data in trials:
            self.__simple_pipeline(clf, 'transform', 'X_new', kwargs, data)


    def test_feature_generation(self):
        # examples based on:
        # http://scikit-learn.org/stable/modules/preprocessing.html
        trials = [(StandardScaler, {}, None),
                  (MinMaxScaler, {}, None),
                  (Normalizer, {}, None),
                  (Binarizer, {'threshold': 0.5}, None)]

        for clf, kwargs, data in trials:
            self.__simple_pipeline(clf, 'transform', 'X_new', kwargs, data)

    def test_moving_params(self):
        digits = datasets.load_digits()
        digits_data = digits.data
        digits_target = digits.target

        p = Pipeline()

        node_data = p.add(NumpyRead(digits_data))
        node_target = p.add(NumpyRead(digits_target))
        node_split = p.add(SplitTrainTest(2, random_state=0))
        # parameters from
        # http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html
        node_clf1 = p.add(
            wrap_and_make_instance(
                RandomForestClassifier,
                max_depth=5,
                n_estimators=10,
                max_features=1,
                random_state=0))
        node_clf2 = p.add(wrap_and_make_instance(RandomForestClassifier, max_depth=12,
                                        n_estimators=100, max_features=1000))
        node_params_out_1 = p.add(CSVWrite(self._tmp_files.get(
            'out_params_1.csv')))
        node_params_out_2 = p.add(CSVWrite(self._tmp_files.get(
            'out_params_2.csv')))
        node_pred_out_1 = p.add(CSVWrite(self._tmp_files.get(
            'out_pred_1.csv')))
        node_pred_out_2 = p.add(CSVWrite(self._tmp_files.get(
            'out_pred_2.csv')))

        node_data['output'] > node_split['input0']
        node_target['output'] > node_split['input1']

        node_split['train0'] > node_clf1['X_train']
        node_split['train1'] > node_clf1['y_train']
        node_split['test0'] > node_clf1['X_test']

        node_split['train0'] > node_clf2['X_train']
        node_split['train1'] > node_clf2['y_train']
        node_split['test0'] > node_clf2['X_test']

        node_clf1['params_out'] > node_clf2['params_in']

        node_clf1['params_out'] > node_params_out_1['input']
        node_clf2['params_out'] > node_params_out_2['input']

        node_clf1['y_pred'] > node_pred_out_1['input']
        node_clf2['y_pred'] > node_pred_out_2['input']

        self.run_pipeline(p)

        params_1 = self._tmp_files.csv_read('out_params_1.csv')
        params_2 = self._tmp_files.csv_read('out_params_2.csv')
        self.assertTrue(np.array_equal(params_1, params_2))

        y_pred_1 = self._tmp_files.csv_read('out_pred_1.csv')
        y_pred_2 = self._tmp_files.csv_read('out_pred_2.csv')
        self.assertTrue(np.array_equal(y_pred_1, y_pred_2))

    def __metric_pipeline(self, metric, params={}, in_data=None):

        X_in, y_in = self.__process_in_data(in_data)

        metric_stage = wrap_and_make_instance(metric, **params)
        in_keys = metric_stage.input_keys
        out_keys = metric_stage.output_keys

        p = Pipeline()

        node_X_in = p.add(NumpyRead(X_in))
        node_y_in = p.add(NumpyRead(y_in))

        node_split = p.add(SplitTrainTest(2, random_state=0))
        node_X_in['output'] > node_split['input0']
        node_y_in['output'] > node_split['input1']

        ctrl_X_train, ctrl_X_test, ctrl_y_train, ctrl_y_test = (
            train_test_split(X_in, y_in, random_state=0))

        node_clf = p.add(wrap_and_make_instance(SVC,
                                       random_state=0))
        node_split['train0'] > node_clf['X_train']
        node_split['train1'] > node_clf['y_train']
        node_split['test0'] > node_clf['X_test']

        ctrl_clf = SVC(random_state=0, probability=True)
        ctrl_clf.fit(ctrl_X_train, ctrl_y_train)

        node_proba_1 = p.add(SplitY(1))
        node_clf['pred_proba'] > node_proba_1['input']

        ctrl_y_score = ctrl_clf.predict_proba(ctrl_X_test)[:, 1]

        node_metric = p.add(metric_stage)

        ctrl_metric_args = {}
        if 'y_true' in in_keys:
            node_split['test1'] > node_metric['y_true']
            ctrl_metric_args['y_true'] = ctrl_y_test
        if 'y_score' in in_keys:
            node_proba_1['y'] > node_metric['y_score']
            ctrl_metric_args['y_score'] = ctrl_y_score
        if 'probas_pred' in in_keys:
            node_proba_1['y'] > node_metric['probas_pred']
            ctrl_metric_args['probas_pred'] = ctrl_y_score

        out_nodes = [p.add(CSVWrite(self._tmp_files('out_{}.csv'.format(
            out_key)))) for out_key in out_keys]
        [node_metric[out_key] > out_nodes[i]['input'] for i, out_key in
         enumerate(out_keys)]

        self.run_pipeline(p)

        ctrl_returns = metric(**ctrl_metric_args)
        if len(out_keys) == 1:
            ctrl_returns = (ctrl_returns,)

        for i, out_key in enumerate(out_keys):
            control = ctrl_returns[i]
            result = self._tmp_files.csv_read(
                    'out_{}.csv'.format(out_key),
                    as_nd=True)
            self.assertTrue(result.shape == control.shape and 
                            np.allclose(result, control))

    def test_metric(self):

        # roc test based on
        # http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html
        iris = datasets.load_iris()
        iris_data = iris.data[iris.target != 2]
        iris_target = iris.target[iris.target != 2]

        in_data = (iris_data, iris_target)

        for metric in [roc_curve, roc_auc_score, precision_recall_curve]:
            self.__metric_pipeline(metric, in_data=in_data)


if __name__ == '__main__':
    unittest.main()
