from sklearn.cross_validation import train_test_split

from upsg.stage import Stage
from upsg.uobject import UObject, UObjectPhase

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
    #TODO wrap sklearn in a more general way, like in wrap_sklearn.wrap
    #TODO split more than one array at a time

    def __init__(self, **kwargs):
        """

        parameters
        ----------
        kwargs: 
            arguments corresponding to the keyword arguments of 
            sklearn.cross_validation.train_test_split 

        """
        self.__kwargs = kwargs

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['train', 'test']

    def run(self, outputs_requested, **kwargs):
        uo_train = UObject(UObjectPhase.Write)
        uo_test = UObject(UObjectPhase.Write)
        in_array = kwargs['in'].to_np()
        train, test = train_test_split(in_array, **self.__kwargs)
        uo_train.from_np(train)
        uo_test.from_np(test)
        return {'train' : uo_train, 'test' : uo_test}
