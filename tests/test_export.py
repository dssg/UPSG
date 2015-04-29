import numpy as np
import os
from os import system
import unittest
import filecmp
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from upsg.wrap.wrap_sklearn import wrap, wrap_and_make_instance
from upsg.uobject import UObject, UObjectPhase
from upsg.pipeline import Pipeline
from upsg.fetch.np import NumpyRead
from upsg.export.plot import Plot
from upsg.export.np import NumpyWrite
from upsg.transform.split import SplitColumn, SplitTrainTest
from upsg.utils import np_nd_to_sa, np_sa_to_nd

from utils import path_of_data, UPSGTestCase


class TestExport(UPSGTestCase):

    def test_plot_roc(self):
        # based on
        # http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html
        from sklearn.svm import SVC
        from sklearn.metrics import roc_curve
        from sklearn import datasets
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
        node_plot = p.add(Plot(self._tmp_files('result.png'), 'co-',
                               title='ROC Curve', xlabel='FPR', ylabel='TPR'))

        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']

        node_split['train0'] > node_clf['X_train']
        node_split['train1'] > node_clf['y_train']
        node_split['test0'] > node_clf['X_test']

        node_clf['pred_proba'] > node_select['in']
        node_select['y'] > node_roc['y_score']
        node_split['test1'] > node_roc['y_true']

        node_roc['fpr'] > node_plot['x']
        node_roc['tpr'] > node_plot['y']

        p.run()
        self.assertTrue(os.path.isfile(self._tmp_files('result.png')))

    def test_numpy_write(self): 
        in_data = np.random.rand(10,10)
        p = Pipeline()
        np_in = p.add(NumpyRead(in_data))
        np_out = p.add(NumpyWrite())
        np_in['out'] > np_out['in']
        p.run()
        self.assertTrue(np.allclose(
            in_data, 
            np_sa_to_nd(np_out.get_stage().result)[0]))

if __name__ == '__main__':
    unittest.main()
