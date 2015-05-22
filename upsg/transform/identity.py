import numpy as np

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd


class Identity(RunnableStage):

    """Passes Uobjects on to further pipeline stages without altering them.
    
    This is mostly useful as the first stage in MetaStages in order to make 
    sure that it takes the correct keys. 
    
    Input keys and output keys are specified on init.

    Exactly 1 of input_keys or output_keys should be specified. The other 
    should be left unspecified

    Parameters
    ----------
    input_keys : list or dict of (str : str) or None
        The names of the input keys that this stage takes. If a list, the 
        output keys will be the same as the input keys but with '_out' 
        appended. If a dictionary, the keys should be the input keys and the 
        values should be the output keys they map to.
    output_keys : list or dict of (str : str) or None
        The names of the output keys that this stage takes. If a list, the 
        input keys will be the same as the input keys but with '_in' appended. 
        If a dictionary, the keys should be the output keys and the values 
        should be the input keys they map to.


    """

    def __init__(self, input_keys=None, output_keys=None):
        if isinstance(input_keys, list) or isinstance(input_keys, tuple):
            self.__input_keys = list(input_keys)
            self.__output_keys = ['{}_out'.format(key) for key in input_keys]
            return
        if isinstance(input_keys, dict):
            self.__input_keys = list(input_keys.keys())
            self.__output_keys = [input_keys[key] for key in self.__input_keys]
            return
        if isinstance(output_keys, list) or isinstance(output_keys, tuple):
            self.__output_keys = list(output_keys)
            self.__input_keys = ['{}_in'.format(key) for key in output_keys]
            return
        if isinstance(output_keys, dict):
            self.__output_keys = list(output_keys.keys())
            self.__input_keys = [output_keys[key] for key in self.__output_keys]
            return
        raise ValueError('list or dict expected for input_keys or for output_keys')

    def get_correspondence(self, in_to_out=True):
        """
        
        returns a dictionary of which input keys correspond to which output keys

        Parameters
        ----------
        in_to_out : bool
            If True, the returned object is a dict of input_key: output key.
            If False, the returned object is a dict of output_key: input_key
        
        """
        if in_to_out:
            return {input_key: output_key for input_key, output_key in 
                    zip(self.__input_keys, self.__output_keys)}
        else:
            return {output_key: input_key for output_key, input_key in 
                    zip(self.__output_keys, self.__input_keys)}

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        correspondence = self.get_correspondence()
        return {correspondence[in_key]: kwargs[in_key] for in_key in kwargs}
