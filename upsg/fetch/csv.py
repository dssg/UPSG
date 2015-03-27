from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase


class CSVRead(RunnableStage):

    def __init__(self, filename):
        self.__filename = filename

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['out']

    def run(self, outputs_requested, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(self.__filename)
        return {'out': uo}
