import abc

from pipeline import Pipeline

class Stage:
    """Base class of all pipeline stages"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_keys(self):
        """A list of keys signifying what this Stage will be expecting as input.

            For some stages, all inputs need not be required. At present, if 
            an input is required but not provided in the pipeline, the user
            will have to deal with a runtime error. In the future, we intend
            to build a way to enforce requirements at the pipeline building
            stage
       
        """
        #TODO enforce required input
        return []

    @abc.abstractproperty
    def output_keys(self):
        """A list signifying the output that this Stage will produce.

        The Stage will be expected to return a dictionary of UObjects where
        each key in the dictionary is the same as one item in output_keys.
        """
        return []

    @abc.abstractmethod
    def run(self, outputs_requested, **kwargs):
        """Run this phase of the pipeline.

        Parameters
        ---------- 
        outputs_requested:
            A list of the output keys that are connected to another Stage of
            the pipeline. A Stage may choose to do less work if some of the
            outputs that it offers will not be used
        kwargs: A collection of keyword arguments corresponding to
            those specified in input_keys. Each argument will provide a 
            readable UObject.

        Returns
        -------
        A dictionary of UObjects that have been written to. The dictionary
        should provide a value of each key specified in output_keys.
        
        """
        return {}
        
class MetaStage(Stage):
    """A Stage that will internally consist of multiple stages connected
    together."""
    
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def pipeline(self):
        """A upsg.Pipeline signifying the subgraph associated with this 
        Metastage"""
        return Pipeline()
