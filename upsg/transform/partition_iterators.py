import numpy as np

from sklearn.cross_validation import _PartitionIterator

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
    sliding mode, the beginning and ending of the training set are both
    incremented by the size of the window in each parition. The example in
    the last paragraph with years used sliding mode. In expanding mode, only
    the end of the training set is incremented, and the training set continues
    to grow with each partition. If we had used expanding mode with the 
    previous example, then our partitions would be: 

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
    init_training_window_start : number
        The lower limit of the values of y used to select the first training
        set. If y is an array of years and we want our first training set
        to cover the years 1999-2000, this variable should be set to 1999
    final_testing_window_end : number
        The upper limit of the values of y used to select the final testing 
        set. If y is an array of years and we want our last testing set to 
        cover the years 2005-2006, this variable should be set to 2006
    window_size : number
        The distance to move the testing window for each partition. Both the
        start and end of the testing window are incremented by window_size.
        If ByWindow is operating in sliding mode, the start and end of the 
        training window will also be incremented by window_size. If ByWindow
        is operating in expanding mode, only the end of the testing window 
        will be incremented by window_size.
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
    >>> init_training_window_start = 1999
    >>> final_testing_window_end = 2006
    >>> window_size = 2
    >>> mode = ByWindowMode.SLIDING
    >>> bw = ByWindow(y, 
    ...               init_training_window_start, 
    ...               final_testing_window_end,
    ...               window_size,
    ...               mode)
    >>> for train_index, test_index in bw:
    ...     print train_index, test_index

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

    @staticmethod
    def n_arrays(y, init_training_window_start,
                 final_testing_window_end, window_size,
                 mode=ByWindowMode.EXPANDING):
        """
        
        Estimates the number of folds (i.e. train/test sets) that will be
        produced given a set of init arguments. This is a consolation to the
        equiment in UPSG, which needs to know this before the class is initialized

        """
        return ((final_testing_window_end - init_training_window_start) / 
                window_size)

    def _iter_test_indices(self):
        window_size = self.__window_size
        train_start = self.__init_training_window_start
        train_end = train_start + window_size - 1
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
            
