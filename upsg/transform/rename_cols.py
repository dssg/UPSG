from copy import deepcopy

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase


class RenameCols(RunnableStage):

    """
    
    Renames columns of the table 'input' and produces a table 'output' 
    that is identical to the 'input' table except with colums renamed .


    Parameters
    ----------
    rename_dict : (dict of str : str) or (list of str)
        If a dict:
            A dictionary mapping old column names to new column names. If the
            table connected to the 'input' input has columns corresponding to the
            keys of this dictionary, the resulting table will have columns
            names with the corresponding values.
        If a list:
            Colums will be renamed in order. The first column will be renamed
            to the first entry of this list. The second column will be renamed
            to the second entry, etc.
    
    """

    def __init__(self, rename_dict):
        self.__rename_dict = rename_dict

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        # TODO maybe we can avoid rewriting all the data (esp in sql) by
        # creating some sort of a "view" object
        uo_out = UObject(UObjectPhase.Write)
        in_array = kwargs['input'].to_np()
        rename_dict = self.__rename_dict

        if isinstance(rename_dict, dict):
            def repl(col):
                try:
                    return rename_dict[col]
                except KeyError:
                    return col
            in_array.dtype.names = map(repl, in_array.dtype.names)
        else:
            in_array.dtype.names = rename_dict
        uo_out.from_np(in_array)

        return {'output': uo_out}
