from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase

from upsg.utils import is_sa, np_nd_to_sa

class NumpyWriteError(Exception):
    pass

class NumpyWrite(RunnableStage):
    """Makes a UObject available as a Numpy array

    **Input Keys**

    input
        table to write to array

    Attributes
    ----------
    Result : numpy.ndarray
        Table that was passed to this Stage

    """

    def __init__(self):
        self.__run = False
        self.__array = None

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return []

    @property
    def result(self):
        if not self.__run:
            raise NumpyWriteError('This stage hasn\'t been run yet')
        return self.__array

    def run(self, outputs_requested, **kwargs):
        self.__array = kwargs['input'].to_np()
        self.__run = True
        return {}
