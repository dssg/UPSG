import numpy as np
from os import system
import unittest
from inspect import getargspec
import string
import random

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
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
from upsg.transform.split import SplitColumn, SplitTrainTest
from upsg.utils import np_nd_to_sa, np_sa_to_nd

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

    def __simple_pipeline(self, sk_cls, sk_method_name, upsg_out_key, 
                          init_kwargs={}, in_data=None):

        if in_data is None:
            in_data = a = np.hstack(
                    (np.random.random((100,10)), 
                     np.random.randint(0, 2, (100,1))))
        elif isinstance(in_data, str) and in_data.split('.')[-1] == 'csv':
            in_data, _ = np_sa_to_nd(csv_read(path_of_data(in_data)))

        p = Pipeline()

        sk_stage = p.add(wrap_and_make_instance(
            sk_cls, 
            **init_kwargs))

        data_in = p.add(NumpyRead(in_data))

        split_y = p.add(SplitColumn(-1))
        data_in['out'] > split_y['in']

        if sk_method_name == 'predict':
            train_test = p.add(SplitTrainTest(2, random_state=0))
            split_y['X'] > train_test['in0']
            split_y['y'] > train_test['in1']

            input_keys = sk_stage.get_stage().input_keys
            if 'X_train' in input_keys:
                train_test['train0'] > sk_stage['X_train']
            if 'X_test' in input_keys:
                train_test['test0'] > sk_stage['X_test']
            if 'y_train' in input_keys:
                train_test['train1'] > sk_stage['y_train']
        else:
            split_y['X'] > sk_stage['X_train']
            split_y['y'] > sk_stage['y_train']

        csv_out = p.add(CSVWrite(self._tmp_files.get('out.csv')))
        sk_stage[upsg_out_key] > csv_out['in']

        p.run()

        ctrl_y = in_data[:, -1]
        ctrl_X = in_data[:, :-1]

        ctrl_sk_inst = sk_cls(**init_kwargs)
        if sk_method_name == 'predict':
            ctrl_X_train, ctrl_X_test, ctrl_y_train, ctrl_y_test = (
                train_test_split(ctrl_X, ctrl_y, random_state=0))
            ctrl_sk_inst.fit(ctrl_X_train, ctrl_y_train)
            control = ctrl_sk_inst.predict(ctrl_X_test)
        else:
            control = ctrl_sk_inst.fit_transform(ctrl_X, ctrl_y)

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
        self.__simple_pipeline(SVC, 'predict', 'y_pred', in_data='numbers.csv')

    def test_factor_selection(self):
        # These are based on the documentation: 
        # http://scikit-learn.org/stable/modules/feature_selection.html
        # and
        # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#example-feature-selection-plot-rfe-digits-py
        vt_in =  np.array([[0, 0, 1, 1], 
                           [0, 1, 0, 0], 
                           [1, 0, 0, 1], 
                           [0, 1, 1, 0], 
                           [0, 1, 0, 1], 
                           [0, 1, 1, 0]])
        iris = datasets.load_iris()
        in_svc = np.hstack(
                (iris.data, iris.target.reshape(iris.target.size, 1)))
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

        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']

        node_split['train0'] > node_clf1['X_train']
        node_split['train1'] > node_clf1['y_train']
        node_split['test0'] > node_clf1['X_test']

        node_split['train0'] > node_clf2['X_train']
        node_split['train1'] > node_clf2['y_train']
        node_split['test0'] > node_clf2['X_test']

        node_clf1['params_out'] > node_clf2['params_in']

        node_clf1['params_out'] > node_params_out_1['in']
        node_clf2['params_out'] > node_params_out_2['in']

        node_clf1['y_pred'] > node_pred_out_1['in']
        node_clf2['y_pred'] > node_pred_out_2['in']

        p.run()

        params_1 = self._tmp_files.csv_read('out_params_1.csv')
        params_2 = self._tmp_files.csv_read('out_params_2.csv')
        self.assertTrue(np.array_equal(params_1, params_2))

        y_pred_1 = self._tmp_files.csv_read('out_pred_1.csv')
        y_pred_2 = self._tmp_files.csv_read('out_pred_2.csv')
        self.assertTrue(np.array_equal(y_pred_1, y_pred_2))

    def testMetric(self):

        # based on
        # http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html
        iris = datasets.load_iris()
        iris_data = iris.data[iris.target != 2]
        iris_target = iris.target[iris.target != 2]

        p = Pipeline()

        node_data = p.add(NumpyRead(iris_data))
        node_target = p.add(NumpyRead(iris_target))
        node_split = p.add(SplitTrainTest(2, random_state=0))
        node_clf = p.add(wrap_and_make_instance(SVC,
                                       random_state=0))
        node_select = p.add(SplitColumn(1))
        node_roc = p.add(wrap_and_make_instance(roc_curve))
        node_fpr_out = p.add(CSVWrite(self._tmp_files.get('out_fpr.csv')))
        node_tpr_out = p.add(CSVWrite(self._tmp_files.get('out_tpr.csv')))

        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']

        node_split['train0'] > node_clf['X_train']
        node_split['train1'] > node_clf['y_train']
        node_split['test0'] > node_clf['X_test']

        node_clf['pred_proba'] > node_select['in']
        node_select['y'] > node_roc['y_score']
        node_split['test1'] > node_roc['y_true']

        node_roc['fpr'] > node_fpr_out['in']
        node_roc['tpr'] > node_tpr_out['in']

        p.run()

        result_fpr = self._tmp_files.csv_read('out_fpr.csv', True)
        result_tpr = self._tmp_files.csv_read('out_tpr.csv', True)

        ctrl_X_train, ctrl_X_test, ctrl_y_train, ctrl_y_test = (
            train_test_split(iris_data, iris_target, random_state=0))
        ctrl_clf = SVC(random_state=0, probability=True)
        ctrl_clf.fit(ctrl_X_train, ctrl_y_train)
        ctrl_y_score = ctrl_clf.predict_proba(ctrl_X_test)[:, 1]
        ctrl_fpr, ctrl_tpr, thresholds = roc_curve(ctrl_y_test, ctrl_y_score)

        self.assertTrue(np.allclose(ctrl_fpr, result_fpr))
        self.assertTrue(np.allclose(ctrl_tpr, result_tpr))

if __name__ == '__main__':
    unittest.main()
