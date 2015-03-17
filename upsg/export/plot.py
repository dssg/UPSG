import matplotlib.pyplot as plt
import numpy as np

from ..stage import RunnableStage
from ..utils import np_sa_to_nd

class Plot(RunnableStage):
    def __init__(self, filename,  *args, 
        **kwargs):
        #TODO documentation
        self.__filename = filename
        try:
            self.__title  = kwargs['title']
            del kwargs['title']
        except KeyError:
            self.__title = ''
        try:
            self.__xlabel = kwargs['xlabel']
            del kwargs['xlabel']
        except KeyError:
            self.__xlabel = ''
        try:
            self.__ylabel = kwargs['ylabel']
            del kwargs['ylabel']
        except KeyError:
            self.__ylabel = ''
        self.__args = args
        self.__kwargs = kwargs
    

    @property
    def input_keys(self):
        return ['x', 'y']

    @property
    def output_keys(self):
        return []

    def run(self, outputs_requested, **kwargs):
        y = np_sa_to_nd(kwargs['y'].to_np())[0]
        try:
            x = np_sa_to_nd(kwargs['x'].to_np())[0]
        except KeyError:
            M = y.shape[0]
            x = np.arange(M).reshape(M, 1)
        plt.plot(x, y, *self.__args, **self.__kwargs)
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.savefig(self.__filename)
        return {}
