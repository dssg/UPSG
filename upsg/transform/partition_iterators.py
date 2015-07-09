import itertools as it
import numpy as np

from sklearn.cross_validation import _PartitionIterator

def by_window_ranges(init_window_start, init_window_end, final_window_end,
                     velocity):
    """Returns an iterator of windows for use as ByWindow input

    Given a starting window, a terminus, and a velocity, generates a sequence
    of windows. This may be used, for example, to generate a sequence of
    start years and end years to use to delimit a training set or a test set
    that is partitioned by year.

    Parameters
    ----------
    init_window_start : number
        The lower boundry of the first window
    init_window_end : number
        The upper boundary of the first window
    final_window_end : number
        The upper boundary of the final window
    velocity : number
        The distance that the upper and lower boundary are moved for each
        subsequent window

    Examples
    --------
    >>> print by_window_ranges(1999, 2000, 2006, 2)
    [(1999, 2000), (2001, 2002), (2003, 2004), (2005, 2006)]
    >>> print by_window_ranges(1, 3, 7, 1)
    [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    """

    return zip(
            xrange(init_window_start, final_window_end, velocity),
            xrange(init_window_end, final_window_end + velocity, velocity))

class ByWindowMode(object):
    EXPANDING, SLIDING = range(2)

class ByWindow(_PartitionIterator):
    """Generates partitions for cross-validation based on some sliding window.

    Implements an interface similar to 
    http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html
    Except it chooses train and test indices that progress through a
    sliding window. For example, the user may wish to test against data 
    generated prior to a certain year and test against data generated after
    that year. Alternatively, the user may wish to train with data generated 
    within a certain distance of a certain location, and test against data 
    generated outside of that distance.
    
    ByWindow will generate a number of such partitions as it slides over a
    certain field. For example, if we are moving over a field that signifies 
    year that a given record was generated, and those years range from 1999 
    to 2006, we may wish to first train on data from 1999-2000 and test on 
    data from 2001-2002, then train on data from 2001-2002 and test on data
    from 2003-2004, then train on data from 2003-2004 and test on data from 
    2005-2006. Each of these partitions will be specified by a set of training
    indices and a set of test indices in the same manner as 
    sklearn.cross_validation.StratifiedKFold. 

    ByWindow can operate in either "sliding" mode or "expanding" mode. In 
    sliding mode, the training set includes only the current training window. 
    The example in the last paragraph with years used sliding mode. 
    In expanding mode, the training set includes the current training window
    and all previous windows. If we had used expanding mode with the previous 
    example, then our partitions would be: 

    1. training on 1999-2000, testing on 2001-2002
    2. training on 1999-2002, testing on 2003-2004
    3. training on 1999-2004, testing on 2005-2006

    Parameters
    ----------
    y : 1-dimensional np.ndarray
        An array of values used to determine which indices are associated with
        each partition. If you want to partition by year, this should be the
        column of your table that signifies year. If you want to partition by
        distance from a point, this is the column of your table that signifies
        distance from a point.
    training_windows : list of (number, number)
        A list of ranges to be used as training windows. For example, 
        ``[(1999, 2000), (2001, 2002), (2003, 2004)]`` specifies 3 windows:
        the first from 1999-2000, the second from 2001-2002, and the third 
        from 2003-2004. All of these boundaries are inclusive.
    testing_windows : list of (number, number)
        A list of ranges to be used as testing windows. For example, 
        ``[(1999, 2000), (2001, 2002), (2003, 2004)]`` specifies 3 windows:
        the first from 1999-2000, the second from 2001-2002, and the third 
        from 2003-2004. All of these boundaries are inclusive.
    mode : {ByWindowMode.SLIDING, ByWindowMode.EXPANDING}
        Mode in which ByWindow is operating. In sliding mode, the start 
        and end of the training set are incremented in each partition.
        In expanding mode, only the end is incremented.

    Examples
    --------
    >>> fines_issued = np.array([(2001, 12.31), (1999, 14.32), (1999, 120.76),
    ...                          (2002, 32.12), (2004, 98.64), (2005, 32.21),
    ...                          (2002, 100.23)],
    ...                         dtype=[('year', int), ('fine', float)])
    >>> y = fines_issued['year']
    >>> mode = ByWindowMode.SLIDING
    >>> training_windows = by_window_ranges(1999, 2000, 2004, 2)
    >>> testing_windows = by_window_ranges(2001, 2002, 2006, 2)
    >>> print training_windows
    [(1999, 2000), (2001, 2002), (2003, 2004)]
    >>> print testing_windows
    [(2001, 2002), (2003, 2004), (2005, 2006)]
    >>> bw = ByWindow(y, 
    ...               training_windows
    ...               testing_windows
    ...               mode)
    >>> for train_index, test_index in bw:
    ...     print 'train_inds:'
    ...     print train_inds
    ...     print 'train_values:'
    ...     print y[train_inds]
    train_inds:
    [1 2 8]
    train_values:
    [1999 1999 2000]
    test_inds:
    [0 3 6]
    test_values:
    [2001 2002 2002]
    train_inds:
    [0 3 6]
    train_values:
    [2001 2002 2002]
    test_inds:
    [4]
    test_values:
    [2004]
    train_inds:
    [4]
    train_values:
    [2004]
    test_inds:
    [5 7]
    test_values:
    [2005 2006]
    """

    def __init__(self, y, training_windows, testing_windows, 
                 mode=ByWindowMode.EXPANDING):
        n = y.shape[0]
        super(ByWindow, self).__init__(n)
        self.__n = n
        self.__y = y
        self.__training_windows = training_windows
        self.__testing_windows = testing_windows
        self.__mode = mode

    @staticmethod
    def est_n_folds(training_windows, testing_windows,
                    mode=ByWindowMode.EXPANDING, *args, **kwargs):
        """
        
        Estimates the number of folds (i.e. train/test sets) that will be
        produced given a set of init arguments. This is a consolation to the
        equiment in UPSG, which needs to know this before the class is 
        initialized

        """
        return min(len(training_windows), 
                   len(testing_windows))

    def _iter_test_indices(self):
        training_windows = self.__training_windows
        testing_windows = self.__testing_windows
        y = self.__y
        mode = self.__mode

        self.__train_mask = np.zeros((self.__n,), dtype=bool)
        for training_window, testing_window in it.izip(training_windows,
                                                       testing_windows):
            new_train_mask = np.logical_and(
                    y >= training_window[0],
                    y <= training_window[1])
            self.__test_mask = np.logical_and(
                    y >= testing_window[0],
                    y <= testing_window[1])
            if mode == ByWindowMode.SLIDING:
                self.__train_mask = new_train_mask
            else:
                self.__train_mask = np.logical_or(self.__train_mask, 
                                                  new_train_mask)
            yield self.__test_mask.nonzero()[0]

    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it's __iter__ so that isn't the case
        for i, (train_index, test_index) in enumerate(
                super(ByWindow, self).__iter__()):
            yield self.__train_mask.nonzero()[0], test_index
            
