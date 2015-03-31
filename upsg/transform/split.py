import tokenize
from StringIO import StringIO
from token import *


from sklearn.cross_validation import train_test_split

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase

class SplitColumns(RunnableStage):
    """Splits a table 'in' into two tables 'selected' and 'rest' where 
    'selected' consists of the given columns and 'rest' consists of the rest
    of the columns"""

    def __init__(self, columns):
        """

        parameters
        ----------
        columns: list of str
            Colums that will appear in the 'selected' table but not the 'rest'
            table
        """
        self.__columns = columns

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['selected', 'rest']

    def run(self, outputs_requested, **kwargs):
        # TODO different implementation if internally sql?
        columns = self.__columns

        to_return = {}
        in_array = kwargs['in'].to_np()

        if 'selected' in outputs_requested:
            uo_selected = UObject(UObjectPhase.Write)
            uo_selected.from_np(in_array[columns])
            to_return['selected'] = uo_selected

        if 'rest' in outputs_requested:
            uo_rest = UObject(UObjectPhase.Write)
            # http://stackoverflow.com/questions/3462143/get-difference-between-two-lists
            remaining_columns = list(set(in_array.dtype.names) - set(columns))
            uo_rest.from_np(in_array[remaining_columns])
            to_return['rest'] = uo_rest

        return to_return


class SplitColumn(RunnableStage):

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
        return {'X': uo_X, 'y': uo_y}


class SplitTrainTest(RunnableStage):

    """Splits a table 'in' into two tables 'train' and 'test' by rows."""
    # TODO wrap.wrap_sklearn in a more general way, like in wrap.wrap_sklearn
    # TODO split more than one array at a time

    def __init__(self, n_arrays=1, **kwargs):
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
        results = {key: UObject(UObjectPhase.Write) for key
                   in self.__output_keys}
        for index, in_key in enumerate(self.__input_keys):
            key_number = int(in_key.replace('in', ''))
            results['train{}'.format(key_number)].from_np(splits[2 * index])
            results['test{}'.format(key_number)].from_np(splits[2 * index + 1])
        return results


class Query(RunnableStage):

    def __init__(self, query):
        self.__query = query

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        # TODO find some interface that doesn't involve string parsing
        # modeled after pandas.Dataframe.query:
        #     http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.query.html
        # which implements its own computation engine:
        #     http://pandas.pydata.org/pandas-docs/dev/generated/pandas.eval.html
        # supports numpy arithmetic comparison operators:
        #     http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#arithmetic-and-comparison-operations
        raise NotImplementedError()
#        result = []
#        g = tokenize.generate_tokens(StringIO(self.__query).readline)
#        for toknum, tokval, _, _, _  in g:
#        if toknum == NUMBER and '.' in tokval:  # replace NUMBER tokens
#            result.extend([
#                (NAME, 'Decimal'),
#                (OP, '('),
#                (STRING, repr(tokval)),
#                (OP, ')')
#            ])
#        else:
#            result.append((toknum, tokval))
#        return untokenize(result)

