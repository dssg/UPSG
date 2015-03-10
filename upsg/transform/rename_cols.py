from copy import deepcopy

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase

class RenameCols(RunnableStage):
    """Renames columns of the table connected to the 'in' input and produces
    a table on the 'out' output that is identical to the 'in' table except with
    colums renamed ."""

    def __init__(self, rename_dict):
        """

        parameters
        ----------
        rename_dict: dict of str : str
            A dictionary mapping old column names to new column names. If the
            table connected to the 'in' input has columns corresponding to the
            keys of this dictionary, the resulting table will have columns 
            names with the corresponding values.

        """
        self.__rename_dict = rename_dict

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        #TODO maybe we can avoid rewriting all the data (esp in sql) by 
        # creating some sort of a "view" object
        uo_out = UObject(UObjectPhase.Write)
        in_array = kwargs['in'].to_np()
        rename_dict = self.__rename_dict
        def repl(col):
            try:
                return rename_dict[col]
            except KeyError:
                return col
        in_array.dtype.names = map(repl, in_array.dtype.names)
        uo_out.from_np(in_array)
        
        return {'out' : uo_out}
        
