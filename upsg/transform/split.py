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
        
