from ..uobject import UObject, UObjectPhase
from ..stage import RunnableStage

from numpy.lib.recfunctions import join_by, rename_fields

class Merge(RunnableStage):
    # TODO we need to support the case where the left table and the right table call the key
    # different things
    """Does an operation analogous to SQL JOIN (or pandas DataFrame.merge)

    Input Keys
    ----------
    in_left : first table to join
    in_right : second table to join
    
    Output Keys
    -----------
    out

    Parameters
    ----------
    left_on : str
        column on which to join the left table
    right_on : str
        column on which to join the right table
    joined_col : str or None
        The name to give the the column on which the tables were joined. If 
        None or not provided, defaults to left_on
    kwargs
        kwargs corresponding to the optional arguments of 
        numpy.lib.recfunctions.join_by
        (http://pyopengl.sourceforge.net/pydoc/numpy.lib.recfunctions.html#-join_by)

    """

    def __init__(self, left_on, right_on, joined_col=None, **kwargs):
        self.__left_on = left_on
        self.__right_on = right_on
        if joined_col:
            self.__joined_col = joined_col
        else:
            self.__joined_col = left_on
        self.__kwargs = kwargs

    @property
    def input_keys(self):
        return ['in_left', 'in_right']
    
    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        in_left = kwargs['in_left'].to_np()
        in_right = kwargs['in_right'].to_np()
        out = UObject(UObjectPhase.Write)
        joined_col = self.__joined_col
        in_left = rename_fields(in_left, {self.__left_on: joined_col})
        in_right = rename_fields(in_right, {self.__right_on: joined_col})
        out.from_np(join_by(joined_col, in_left, in_right, **self.__kwargs))
        return {'out': out}
