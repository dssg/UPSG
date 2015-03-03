from sklearn.cross_validation import train_test_split

from ..stage import Stage
from ..uobject import UObject, UObjectPhase

class SplitColumn(Stage):
    """Splits a table 'in' into two tables 'X' and 'y' where y is one column of
    A and X is everything else. """

    def __init__(self, column):
        """

        parameters
        ----------
        column: int or str
            index or name of the column from which 'y' will be extracted

        """
        self.__column = column

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['X', 'y']

    def run(self, outputs_requested, **kwargs):
        uo_X = UObject(UObjectPhase.Write)
        uo_y = UObject(UObjectPhase.Write)
        in_array = kwargs['in'].to_np()
        names = list(in_array.dtype.names)
        if isinstance(self.__column, int):
            col_name = names[self.__column]
        else:
            col_name = self.__column
        uo_y.from_np(in_array[[col_name]])
        names.remove(col_name)
        uo_X.from_np(in_array[names])
        return {'X' : uo_X, 'y' : uo_y}
        
class SplitTrainTest(Stage):
    """Splits a table 'in' into two tables 'train' and 'test' by rows."""
    #TODO wrap.wrap_sklearn in a more general way, like in wrap.wrap_sklearn
    #TODO split more than one array at a time

    def __init__(self, n_arrays = 1, **kwargs):
        """

        parameters
        ----------
        n_arrays: int (default 1)
            the number of arrays that will be split
        kwargs:
            arguments corresponding to the keyword arguments of 
            sklearn.cross_validation.train_test_split 

        """
        self.__kwargs = kwargs
        self.__n_arrays = n_arrays

        self.__input_keys = map('in{}'.format, xrange(n_arrays)) 
        self.__output_keys = (map('train{}'.format, xrange(n_arrays)) +
            map('test{}'.format, xrange(n_arrays)))

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        in_arrays = [kwargs[key].to_np() for key in self.__input_keys]
        splits = train_test_split(*in_arrays, **self.__kwargs)
        results = {key : UObject(UObjectPhase.Write) for key 
            in self.__output_keys}
        for index, in_key in enumerate(self.__input_keys):
            key_number = int(in_key.replace('in', ''))
            print '+++++' + str(index)
            results['train{}'.format(key_number)].from_np(splits[2 * index])
            results['test{}'.format(key_number)].from_np(splits[2 * index + 1])
        return results
