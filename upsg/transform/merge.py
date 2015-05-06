from ..uobject import UObject, UObjectPhase
from ..stage import RunnableStage

from numpy.lib.recfunctions import join_by

class Merge(RunnableStage):
    """Does an operation analogous to SQL JOIN (or pandas DataFrame.merge)

    Input Keys
    ----------
    in0 : first table to join
    in1 : second table to join
    
    Output Keys
    -----------
    out

    Parameters
    ----------
    cols : str or list of str
        the name of the column to join on or a list of column to join on
    kwargs
        kwargs corresponding to the optional arguments of 
        numpy.lib.recfunctions.join_by
        (http://pyopengl.sourceforge.net/pydoc/numpy.lib.recfunctions.html#-join_by)

    """

    def __init__(self, cols, **kwargs):
        self.__cols = cols
        self.__kwargs = kwargs

    @property
    def input_keys:
        return ['in0', 'in1']
    
    @property
    def output_keys:
        return ['out']

    def run(self, outputs_requested, **kwargs):
        in0 = kwargs['in0'].to_np()
        in1 = kwargs['in1'].to_np()
        out = UObject(UObjectPhase.Write)
        out.from_np(join_by(self.__cols, in0, in1, **self.__kwargs))
        return {'out': out}
