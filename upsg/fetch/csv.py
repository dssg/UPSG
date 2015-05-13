from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase


class CSVRead(RunnableStage):
    """Stage to read in a csv

    **Output Keys**
        
    Output
        table read from csv

    Parameters
    ----------
    filename : str
        filename of the csv
    kwargs : dict
        keyword arguments to pass to numpy.genfromtxt
        (http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html)
        If no kwargs are provided, we use: dtype=None, delimiter=',', 
        names=True.

    """

    def __init__(self, filename, **kwargs):
            
        self.__filename = filename
        self.__kwargs = kwargs

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(self.__filename, **self.__kwargs)
        return {'output': uo}
