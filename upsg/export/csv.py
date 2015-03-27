from ..stage import RunnableStage


class CSVWrite(RunnableStage):

    def __init__(self, filename):
        self.__filename = filename

    def __repr__(self):
        return 'CSVWrite({})'.format(self.__filename)

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return []

    def run(self, outputs_requested, **kwargs):
        kwargs['in'].to_csv(self.__filename)
        return {}
