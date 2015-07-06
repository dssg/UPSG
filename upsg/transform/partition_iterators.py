import numpy as np

from sklearn.cross_validation import _PartitionIterator

"""Generates partitions for cross-validation based on time

Implements an interface similar to 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html
Except it chooses train and test indices that progress through time. It will 
begin training on the earliest unique time and testing on the second earliest
unique time, then it will train on the earliest and second earliest unique 
time and test on the third earliest unique time, then it will train on the
three earliest unique times and test on the 4th earliest, etc.

Parameters
----------
y : 1-dimensional np.ndarray
    An array of times. Indices will be chosen based on the times in this
    array
n_folds : int
    Number of partitions to generate. If there are not enough unique times
    in the given y, then fewer than n_folds partitions will be returned

Examples
--------
>>> times = np.array([2010, 2009, 2010, 2012, 2009, 2014, 2015])
>>> n_folds = 3

>>> temporal = Temporal(times, n_folds)
>>> for train_index, test_index in temporal:
    ...    print train_index, test_index

    [1, 4] [0 2]
    [0, 1, 2, 4] [3]
    [0, 1, 2, 3, 4] [5]

"""

class Temporal(_PartitionIterator):
    def __init__(self, y, n_folds=3):
        n = y.shape[0]
        super(Temporal, self).__init__(n)
        self.__n = n
        self.__y = y
        self.__n_folds = n_folds
    def _iter_test_indices(self):
        unique_years = np.unique(self.__y)
        self.__train_mask = np.zeros(self.__y.shape, dtype=bool)
        self.__test_mask = self.__y == unique_years[0]
        for test_year in unique_years[1:]:
            self.__train_mask = np.logical_or(self.__train_mask, 
                                              self.__test_mask)
            self.__test_mask = self.__y == test_year
            yield np.where(self.__test_mask)
    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it's __iter__ so that isn't the case
        for i, (train_index, test_index) in enumerate(
                super(Temporal, self).__iter__()):
            if i >= self.__n_folds:
                break;
            yield self.__train_mask.nonzero()[0].tolist(), test_index
            
