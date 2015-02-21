from upsg.stage import Stage
from upsg.UObject import UObject, UObjectPhase

class CSVRead(Stage):
    def __init__(self, filename):
        self.__filename = filename
    
    @property
    def input_keys(self):
        return {}

    @property
    def output_keys(self):
        return ['out']

    def run(self, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(self.__filename)
        return {'out' : uo}
