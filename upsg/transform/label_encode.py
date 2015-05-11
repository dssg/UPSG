import numpy as np

from sklearn.preprocessing import LabelEncoder

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd


class LabelEncode(RunnableStage):

    """
    
    Encodes all strings to a value with behavior similar to 
    sklearn.preprocessing.LabelEncoder
    
    """

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        uo_out = UObject(UObjectPhase.Write)
        in_array = kwargs['input'].to_np()
        le = LabelEncoder()
        new_dtype = []
        result_arrays = []
        for (col_name, fmt) in in_array.dtype.descr:
            if 'S' in fmt:
                result_arrays.append(le.fit_transform(in_array[col_name]))
                new_dtype.append((col_name, int))
            else:
                result_arrays.append(in_array[col_name])
                new_dtype.append((col_name, fmt))
        out_array = np.array(zip(*result_arrays), dtype=new_dtype)
        uo_out.from_np(out_array)
        return {'output': uo_out}
