import numpy as np

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd


class Identity(RunnableStage):

    """Passes Uobjects on to further pipeline stages without altering them.
    
    This is mostly useful as the first stage in MetaStages in order to make sure that it takes
    the correct keys. 
    
    Input keys and output keys are specified on init.

    """

    def __init__(self, input_keys):
        """

        Parameters
        ----------
        input_keys list or dict of (str: str)
            The names of the input keys that this stage takes. If a list, the output keys will be
            the same as the input keys but with '_output' appended. If a dictionary, the keys 
            should be the input keys and the values should be the output keys they map to.

        """
        if isinstance(input_keys, list) or isinstance(input_keys, tuple):
            self.__input_keys = input_keys
            self.__output_keys = ['{}_out'.format(key) for key in input_keys]
            return
        if isinstance(input_keys, dict):
            self.__input_keys = input_keys.keys()
            self.__output_keys = [input_keys[key] for key in self.__input_keys]
            return
        raise ValueError('list or dict expected for input_keys')

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        return {out_key: kwargs[in_key] for in_key, out_key in 
                zip(self.__input_keys, self.__output_keys)}
