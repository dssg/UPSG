from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase

from upsg.utils import is_sa, np_nd_to_sa


class NumpyRead(RunnableStage):
    """Read into pipeline from numpy array

    **Output Keys**

    output
        table read from Numpy array

    Parameters
    ----------
    A : numpy.ndarray
        Numpy array to read in

    """

    def __init__(self, A):
        self.__A = A

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_np(self.__A)
        return {'output': uo}
