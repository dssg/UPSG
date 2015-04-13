import unittest
import pickle
from os import system
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

from upsg.fetch.np import NumpyRead
from upsg.wrap.wrap_sklearn import wrap_and_make_instance
from upsg.export.csv import CSVWrite
from upsg.transform.split import SplitTrainTest
from upsg.pipeline import Pipeline

from utils import path_of_data, UPSGTestCase


class TestSKLearnParity(UPSGTestCase):

    def test_tutorial(self):
        """

        Verifies we can do what sklearn does here:
        http://scikit-learn.org/stable/tutorial/basic/tutorial.html

        """
        digits = datasets.load_digits()
        digits_data = digits.data
        # for now, we need a column vector rather than an array
        digits_target = digits.target

        p = Pipeline()

        # load data from a numpy dataset
        stage_data = NumpyRead(digits_data)
        stage_target = NumpyRead(digits_target)

        # train/test split
        stage_split_data = SplitTrainTest(2, test_size=1, random_state=0)

        # build a classifier
        stage_clf = wrap_and_make_instance(SVC, gamma=0.001, C=100.)

        # output to a csv
        stage_csv = CSVWrite(self._tmp_files('out.csv'))

        node_data, node_target, node_split, node_clf, node_csv = map(
            p.add, [
                stage_data, stage_target, stage_split_data, stage_clf,
                stage_csv])

        # connect the pipeline stages together
        node_data['out'] > node_split['in0']
        node_target['out'] > node_split['in1']
        node_split['train0'] > node_clf['X_train']
        node_split['train1'] > node_clf['y_train']
        node_split['test0'] > node_clf['X_test']
        node_clf['y_pred'] > node_csv['in']

        p.run()

        result = self._tmp_files.csv_read('out.csv', True)

        # making sure we get the same result as sklearn
        clf = SVC(gamma=0.001, C=100.)
        # The tutorial just splits using array slicing, but we need to make
        #   sure that both UPSG and sklearn are splitting the same way, so we
        #   do something more sophisticated
        train_X, test_X, train_y, test_y = train_test_split(
            digits_data, digits_target, test_size=1, random_state=0)
        clf.fit(train_X, np.ravel(train_y))
        control = clf.predict(test_X)[0]

        self.assertAlmostEqual(result, control)

        # model persistance
        s = pickle.dumps(stage_clf)
        stage_clf2 = pickle.loads(s)
        self.assertEqual(stage_clf.get_params(), stage_clf2.get_params())

if __name__ == '__main__':
    unittest.main()
