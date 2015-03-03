from upsg.stage import Stage
from upsg.uobject import UObject, UObjectPhase

class FromNumpy(Stage):
    def __init__(self, A):
        self.__A = A
    
    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_np(self.__A)
        return {'out' : uo}
