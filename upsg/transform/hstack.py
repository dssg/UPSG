from numpy.lib.recfunctions import merge_arrays

from ..uobject import UObject, UObjectPhase
from ..stage import RunnableStage


class HStack(RunnableStage):
    """Stacks two tables column-wise in a manner similar to numpy.hstack

    **Input Keys**

    input0, input1, ...
        tables to be stacked. Must have the same number of rows
    
    **Output Keys**

    output

    Parameters
    ----------
    n_arrays : int
        Number of arrays being input

    """

    def __init__(self, n_arrays):
        self.__n_arrays = n_arrays
        self.__input_keys = ['input{}'.format(i) for i in xrange(n_arrays)]

    @property
    def input_keys(self):
        return self.__input_keys
    
    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        arrays = [kwargs[input_key].to_np() for input_key in 
                  self.__input_keys]
        # http://stackoverflow.com/questions/15815854/how-to-add-column-to-numpy-array
        out = UObject(UObjectPhase.Write)
        out.from_np(merge_arrays(arrays, flatten=True))
        return {'output': out}
