import unittest
import pickle
from os import system
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold as SKKFold

from upsg.fetch.np import NumpyRead
from upsg.wrap.wrap_sklearn import wrap
from upsg.export.csv import CSVWrite
from upsg.transform.split import SplitTrainTest
from upsg.pipeline import Pipeline
from upsg.model.grid_search import GridSearch
from upsg.model.cross_validation import CrossValidationScore
from upsg.utils import np_sa_to_dict

from utils import path_of_data, UPSGTestCase


class TestModel(UPSGTestCase):

    def xtest_grid_search(self):
        """

        Simulates behavior of example in:
        http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

        """
        from sklearn.svm import SVC

        parameters = {
            'kernel': (
                'rbf',
                'linear'),
            'C': [
                1,
                10],
            'random_state': [0]}
        iris = datasets.load_iris()
        iris_data = iris.data
        iris_target = np.array([iris.target]).T

        p = Pipeline()

        node_data = p.add(NumpyRead(iris_data))
        node_target = p.add(NumpyRead(iris_target))
        node_split = p.add(SplitTrainTest(2, random_state=1))
        node_search = p.add(GridSearch(wrap(SVC), 'score', parameters))
        node_params_out = p.add(CSVWrite(self._tmp_files.get('out.csv')))

        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']
        node_split['train0'] > node_search['X_train']
        node_split['train1'] > node_search['y_train']
        node_split['test0'] > node_search['X_test']
        node_split['test1'] > node_search['y_test']
        node_search['params_out'] > node_params_out['in']

        p.run()

        control = {'kernel': 'linear', 'C': 1, 'random_state': 0}
        result = self._tmp_files.csv_read('out.csv')
        self.assertEqual(np_sa_to_dict(np.array([result])), control)

    def test_cross_validation_score(self):
        rows = 100
        folds = 10

        X = np.random.random((rows, 10))
        y = np.random.randint(0, 2, (rows))
        
        p = Pipeline()

        np_in_X = p.add(NumpyRead(X))
        np_in_y = p.add(NumpyRead(y))

        cv_score = p.add(CrossValidationScore(wrap(SVC), 'score', {}, folds,
                                              random_state=0))               
        np_in_X['out'] > cv_score['X_train']
        np_in_y['out'] > cv_score['y_train']

        score_out = p.add(CSVWrite(self._tmp_files('out.csv')))
        cv_score['score'] > score_out['in']

        p.run()

        result = self._tmp_files.csv_read('out.csv')['f0']

        ctrl_kf = SKKFold(rows, folds, random_state=0)
        ctrl = np.mean(cross_val_score(SVC(), X, y, cv=ctrl_kf))

        self.assertTrue(np.allclose(ctrl, result))

if __name__ == '__main__':
    unittest.main()
