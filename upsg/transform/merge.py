from ..uobject import UObject, UObjectPhase
from ..stage import RunnableStage


class Merge(RunnableStage):
    # TODO we need to support the case where the left table and the right table call the key
    # different things
    """Does an operation analogous to SQL JOIN (or pandas DataFrame.merge)

    Input Keys
    ----------
    input_left : first table to join
    input_right : second table to join
    
    Output Keys
    -----------
    output

    Parameters
    ----------
    left_on : str
        column on which to join the left table
    right_on : str
        column on which to join the right table
    kwargs
        kwargs corresponding to the optional arguments of 
        pandas.DataFrame.merge other than left_on and right_on

    """

    def __init__(self, left_on, right_on, **kwargs):
        self.__left_on = left_on
        self.__right_on = right_on
        self.__kwargs = kwargs

    @property
    def input_keys(self):
        return ['input_left', 'input_right']
    
    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        in_left = kwargs['input_left'].to_dataframe()
        in_right = kwargs['input_right'].to_dataframe()
        out = UObject(UObjectPhase.Write)
        out.from_dataframe(in_left.merge(
            in_right, 
            left_on=self.__left_on, 
            right_on=self.__right_on,
            **self.__kwargs))
        return {'output': out}
