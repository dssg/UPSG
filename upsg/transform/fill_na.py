import numpy as np

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd


class FillNA(RunnableStage):

    """Fills NaNs with some default value"""

    def __init__(self, default_value):
        """

        parameters
        ----------
        default_value: number

        """
        self.__default_value = default_value

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        default_value = self.__default_value
        uo_out = UObject(UObjectPhase.Write)
        in_array = kwargs['input'].to_np()
        # http://stackoverflow.com/questions/5124376/convert-nan-value-to-zero
        for (col_name, fmt) in in_array.dtype.descr:
            if 'f' in fmt:
                in_array[col_name][np.isnan(in_array[col_name])] = default_value 
        uo_out.from_np(in_array)

        return {'out': uo_out}
