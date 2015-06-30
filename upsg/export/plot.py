import matplotlib.pyplot as plt
import numpy as np

from ..stage import RunnableStage
from ..utils import np_sa_to_nd
from ..uobject import UObject, UObjectPhase


class Plot(RunnableStage):
    """Stage to make a plot

    **Input Keys**

    x
        table of x coords
    y
        table of y coords

    **Output Keys**

    plot_file
        the name of the file to which the plot was saved

    Parameters
    ----------
    file_name: str
        The name of the file to which the plot will be saved
    title: str
        Title of the plot
    xlabel: str
        Label of the x-axis
    ylabel: str
        Label of the y-axis
    args: list
        additional args to pass to matplotlib.pyplot.plot
    kwargs: dict
        additional kwargs to pass to matplotlib.pyplot.plot


    """

    def __init__(self, file_name, *args,
                 **kwargs):
        self.__file_name = file_name
        try:
            self.__title = kwargs['title']
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
        return ['plot_file']

    def run(self, outputs_requested, **kwargs):
        y = kwargs['y'].to_np()
        try:
            x = np_sa_to_nd(kwargs['x'].to_np())[0]
        except KeyError:
            M = y.shape[0]
            x = np.arange(M).reshape(M, 1)
        y_nd = np_sa_to_nd(y)[0]
        plt.plot(x, y_nd, *self.__args, **self.__kwargs)
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.legend(y.dtype.names)
        plt.savefig(self.__file_name)
        plt.close()
        uo_plot = UObject(UObjectPhase.Write)
        uo_plot.from_external_file(self.__file_name)
        return {'plot_file': uo_plot}
