import numpy as np

from sklearn.cross_validation import _PartitionIterator

class Temporal(_PartitionIterator):
    def __init__(self, y, n_folds=3):
        # y is the time column
        # for example, say y = np.array([2010, 2009, 2010, 2012, 2009, 2014, 2015])
        # n_folds = 3.
        # First iteration: train is 2009, test is 2010, so indices are:
        # test = [1, 4]
        # train = [0, 2]
        # Second iteration, train is 2009, 2010, test is 2012
        # test = [0, 1, 2, 4]
        # train = [3]
        # Third iteration, train is 2009-2012, test is 2014
        # test = [0, 1, 2, 3, 4]
        # train = [5]
        n = y.shape[0]
        super(Temporal, self).__init__(self, n)
        self.__n = n
        self.__y = y
        self.__n_folds = n_folds
    def _iter_test_indices(self):
        unique_years = np.unique(self.__y)
        self.__train_mask = np.zeros(self.__y.shape, dtype=bool)
        self.__test_mask = y == unique_years[0]
        for test_year in unique_years[1:]:
            self.__train_mask = np.logical_or(self.__train_mask, 
                                              self.__test_mask)
            self.__test_mask = y == test_year
            yield np.where(self.__test_mask)
    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it so that isn't the case
        for train_index, test_index in super(Temporal, self).__iter__(self):
            yield np.where(self.__train_mask), test_index
            
