from upsg.stage import Stage
from upsg.uobject import UObject, UObjectPhase, UObjectException

class CSVRead(Stage):
    def __init__(self, uo):
        if uo.get_phase() != UObjectPhase.Write or not uo.is_finalized():
            raise UObjectException(('Provided UObject must be finalized '
                'and in the Write phase'))
        self.__uo = uo
    
    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        return {'out' : self.__uo}
