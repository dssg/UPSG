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
        return ['in']

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        # TODO maybe we can avoid rewriting all the data (esp in sql) by
        # creating some sort of a "view" object
        uo_out = UObject(UObjectPhase.Write)
        in_array = kwargs['in'].to_np()
        nd_view = np_sa_to_nd(in_array)[0]
        # http://stackoverflow.com/questions/5124376/convert-nan-value-to-zero
        nd_view[np.isnan(nd_view)] = self.__default_value
        uo_out.from_np(in_array)

        return {'out': uo_out}
