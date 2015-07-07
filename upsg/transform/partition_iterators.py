import numpy as np

from sklearn.cross_validation import _PartitionIterator

class ByWindowMode(object):
    EXPANDING, SLIDING = range(2)

class ByWindow(_PartitionIterator):
    """Generates partitions for cross-validation based on some sliding window.

    Implements an interface similar to 
    http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html
    Except it chooses train and test indices that progress through a
    sliding window. For example, It 
    will begin training on the earliest unique time and testing on the second 
    earliest unique time, then it will train on the earliest and second 
    earliest unique time and test on the third earliest unique time, then it 
    will train on the three earliest unique times and test on the 4th earliest,
    etc.

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
    [1, 4] [0, 2]
    [0, 1, 2, 4] [3]
    [0, 1, 2, 3, 4] [5]

    """

    

    def __init__(self, y, init_training_window_start,
                 final_testing_window_end, window_size,
                 mode=ByWindowMode.EXPANDING):
                 
        n = y.shape[0]
        super(ByWindow, self).__init__(n)
        self.__n = n
        self.__y = y
        self.__init_training_window_start = init_training_window_start
        self.__window_size = window_size
        self.__final_testing_window_end = final_testing_window_end
        self.__mode = mode

    def _iter_test_indices(self):
        window_size = self.__window_size
        train_start = self.__init_training_window_start
        train_end = train_start + window_size
        test_end = train_end + window_size
        test_terminate = self.__final_testing_window_end
        y = self.__y
        mode = self.__mode

        while test_end <= test_terminate:
            self.__train_mask = np.logical_and(
                    y >= train_start,
                    y <= train_end)
            self.__test_mask = np.logical_and(
                    y > train_end,
                    y <= test_end)
            if mode == ByWindowMode.SLIDING:
                train_start = train_end
            train_end = test_end
            test_end += window_size
            yield self.__test_mask.nonzero()[0]

    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it's __iter__ so that isn't the case
        for i, (train_index, test_index) in enumerate(
                super(ByWindow, self).__iter__()):
            yield self.__train_mask.nonzero()[0], test_index
            
