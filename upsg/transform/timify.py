
from datetime import strptime


class Timify(RunnableStage):
    """Transforms string columns that look like dates into datetime64 columns

    Strings must follow ISO 8601 time or datetime format in accordance with:
    (http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html)

    Input Keys
    ----------
    in

    Output Keys
    -----------
    out

    """

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['out']


    def run(self, outputs_requested, **kwargs):
        in_data = kwargs['in'].to_np()
        cols = []
        dtype = []
        
        for name, sub_dtype in_data.dtype.descr:
            col = in_data[name]
            if 'S' in sub_dtype:
                try:
                    col = col.astype('M8')
                    sub_dtype = 'M8'
                except ValueError: # not a time
                    pass
            cols.append(col)
            dtype.append((name, sub_dtype))

        uo_out = UObject(UObjectPhase.Write)
        uo_out.from_np(np.array(it.izip(*cols), dtype=dtype))
        return {'out': uo_out}

        
        
