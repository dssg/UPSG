import matplotlib.pyplot as plt
import numpy as np

from ..stage import RunnableStage

class Plot(RunnableStage):
    def __init__(self, filename, title = '', xlabel = '', ylabel = '', *args, 
        #TODO documentation
        **kwargs):
        self.__filename = filename
        self.__title  = title
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__args = args
        self.__kwargs = kwargs
    

    @property
    def input_keys(self):
        return ['x', 'y']

    @property
    def output_keys(self):
        return []

    def run(self, outputs_requested, **kwargs):
        y = kwargs['y'].to_np()
        try:
            x = kwargs['x'].to_np()
        except KeyError:
            M = y.shape[0]
            x = np.arange(M).reshape(M, 1)
        plt.plot(x, y, *self.__args, **self.__kwargs)
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.savefig(self.__filename)
        return {}
