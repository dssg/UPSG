from ..stage import RunnableStage


class CSVWrite(RunnableStage):
    """Write table to csv

    **Input Keys**

    input
        table to write to csv

    Parameters
    ----------
    filename : str
        Name of csv file to write to


    """

    def __init__(self, filename):
        self.__filename = filename

    def __repr__(self):
        return 'CSVWrite({})'.format(self.__filename)

    @property
    def input_keys(self):
        return ['input']

    @property
    def output_keys(self):
        return []

    def run(self, outputs_requested, **kwargs):
        kwargs['input'].to_csv(self.__filename)
        return {}
