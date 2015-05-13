import numpy as np
import itertools as it

from upsg.uobject import UObject, UObjectPhase
from upsg.stage import RunnableStage

class Timify(RunnableStage):
    """Transforms string columns that look like dates into datetime64 columns

    Strings must follow ISO 8601 time or datetime format in accordance with:
    (http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html)

    **Input Keys**

    input

    **Output Keys**
    
    output

    """

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        in_data = kwargs['input'].to_np()
        cols = []
        dtype = []
        
        for name, sub_dtype in in_data.dtype.descr:
            col = in_data[name]
            if 'S' in sub_dtype:
                try:
                    col = col.astype('M8')
                    sub_dtype = col.dtype
                except ValueError: # not a time
                    pass
            cols.append(col)
            dtype.append((name, sub_dtype))

        uo_out = UObject(UObjectPhase.Write)
        uo_out.from_np(np.fromiter(it.izip(*cols), dtype=dtype))
        return {'output': uo_out}

        
        
