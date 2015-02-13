from abs import ABCMeta

class Stage:
    """Base class of all pipeline stages"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def input_keys:
        """A dictionary signifying what this Stage will be expecting as input.
       
        Each key in the dictionary should be the name of a keyword in the 
        keyword arguments will be provided as input. The value of the key
        should be a boolean signifying whether or not the argument is optional.

        """
        return {}

    @abstractproperty
    def output_keys:
        """A list signifying the output that this Stage will produce.

        The Stage will be expected to return a dictionary of UObjects where
        each key in the dictionary is the same as one item in output_keys.
        """
        return []

    @abstractmethod
    def run(self, **kwargs):
        """Run this phase of the pipeline.

        Parameters
        ---------- 
        kwargs: A collection of keyword arguments corresponding to
            those specified in input_keys. Each argument will provide a 
            readable UObject.

        Returns
        -------
        A dictionary of UObjects that have been written to. The dictionary
        should provide a value of each key specified in output_keys.
        
        """
        return {}
        