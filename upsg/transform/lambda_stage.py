import inspect
import numpy as np

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd

class LambdaStage(RunnableStage):
    """Execute arbitrary instructions on Numpy structured arrays

    Allows the user to pass in an arbitrary function which operates on
    0 or more Numpy structured arrays corresponding to UPSG tables and
    returns either a Numpy structured array or a tuple of structured
    arrays or None.

    Input Keys
    ----------
    Input keys will correspond to the function used to initialize
    LambdaStage. For example, if you initialize a LambdaStage with: 

    >>> my_func = lambda employees, hours: return do_custom_join(
    ...     employees, hours)
    >>> stage = LambdaStage(my_func, ['result'])

    Then the input keys will be 'employees' and 'hours'

    Output Keys
    -----------
    Output keys will correspond to the output_keys argument passed to 
    initialize LambdaStage. For example, if you initialize LambdaStage with:

    >>> my_func = lambda to_split: return (to_split[:50], to_split[50:])
    >>> stage = LambdaStage(my_func, ['first_fifty', 'rest'])

    Then the output keys will be 'first_fifty' and 'rest'

    """


    def __init__(self, func, output_keys):
        """

        Parameters
        ----------
        func: (0 or more structured arrays) -> 
            (structured array or tuple of structured arrays or None)
        
            Function to apply at this Stage. Numpy structured arrays will
            be passed to the function and the function is expected to output
            either one structured array if output_keys has length 1,
            a tuple of structured arrays if output_keys has a length of more
            than 1, or None if output_keys has a length of 0
       
        output_keys: tuple of str:
            
            The keys to which outputs will be assigned. If func returns a
            structured array, output_keys should be a tuple of length 1. If 
            func returns a tuple of structured arrays, output_keys should have
            length equal to the length of the returned tuple. If func returns
            None, output_keys should be a tuple of length 0

        """

        self.__func = func
        self.__input_keys = inspect.getargspec(func).args
        self.__n_results = len(output_keys)
        self.__output_keys = output_keys

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        fxs = self.__func(**{key: kwargs[key].to_np()
                            for key in kwargs})
        if self.__n_results == 0:
            fxs = []
        if self.__n_results == 1:
            fxs = [fxs]
        ret = {key: UObject(UObjectPhase.Write) for key in self.__output_keys}
        [ret[key].from_np(fxs[i])
            for i, key in enumerate(self.__output_keys)]
        return ret

