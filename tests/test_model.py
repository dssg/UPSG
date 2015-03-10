import unittest
import pickle
from os import system
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

from upsg.fetch.np import NumpyRead
from upsg.wrap.wrap_sklearn import wrap
from upsg.export.csv import CSVWrite
from upsg.transform.split import SplitTrainTest
from upsg.pipeline import Pipeline
from upsg.model.param_sweep import ParamSweep

from utils import path_of_data

outfile_name = path_of_data('_out.csv')

class TestModel(unittest.TestCase):
    def test_param_sweep(self):
        """

        Simulates behavior of:
        http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

        """
        #>>> from sklearn import svm, grid_search, datasets
        #>>> iris = datasets.load_iris()
        #>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        #>>> svr = svm.SVC()
        #>>> clf = grid_search.GridSearchCV(svr, parameters)
        #>>> clf.fit(iris.data, iris.target)

        from sklearn.svm import SVC

        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        iris = datasets.load_iris()
        iris_data = iris.data
        iris_target = np.array([iris.target]).T
        
        p = Pipeline()

        node_data = p.add(NumpyRead(iris_data))
        node_target = p.add(NumpyRead(iris_target))
        node_split = p.add(SplitTrainTest(2, random_state = 0))
        node_search = p.add(ParamSweep(wrap(SVC), 'score', parameters))
        node_params_out = p.add(CSVWrite(outfile_name))

        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']
        node_split['train0'] > node_search['X_train']
        node_split['train1'] > node_search['y_train']
        node_split['test0'] > node_search['X_test']
        node_split['test1'] > node_search['y_test']
        node_search['params_out'] > node_params_out['in']

        p.run()


        
    def tearDown(self):
        system('rm *.upsg')
        system('rm {}'.format(outfile_name))

if __name__ == '__main__':
    unittest.main()
