from .stage import Stage

in_attrs = ('fit', 'transform')
out_attrs = ('transform', 'predict')

def wrap(skt_class):
    """Wraps a scikit BaseEstimator class inside a UPSG Stage and returns it
    """
    class Wrapped(Stage):
        __skt_class = skt_class
        #TODO stub
        pass
    return Wrapped
