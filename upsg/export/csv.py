from upsg.stage import Stage

class CSVWrite(Stage):
    def __init__(self, filename):
        self.__filename = filename
    
    @property
    def input_keys(self):
        return {'in'}

    @property
    def output_keys(self):
        return []

    def run(self, **kwargs):
        kwargs['in'].to_csv(self.__filename)
        return {}
