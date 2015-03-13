import numpy as np

def np_sa_to_nd(sa):
    """Returns a view of a numpy structured array as a single-type 1 or 
    2-dimensional array. If the resulting nd array would be a column vector,
    returns a 1-d array instead. If the resulting array would have a single 
    entry, returns a 0-d array instead

    At present, uses the 0th element of the 0th column to determine the 
    datatype of the array. 

    Based on http://wiki.scipy.org/Cookbook/Recarray.

    Parameters
    ----------
    sa: numpy.ndarray
        The structured array to view
    
    Returns
    -------
    A tuple (nd, dtype)
        where nd is numpy.ndarray array view and dtype is the numpy.dtype of 
        the structured array that was passed in.
    """
    #TODO use a better metric to determine the datatype
    view = sa.view(dtype = sa[0][0].dtype)
    if len(sa.dtype) == 1:
        if len(sa) == 1:
            return (view.reshape(()), sa.dtype)
        return (view.reshape(len(sa)), sa.dtype)
    return (view.reshape(len(sa), -1), sa.dtype)

def np_nd_to_sa(nd, dtype = None):
    """Returns a view of a numpy, single-type, 0, 1 or 2-dimensional array as a 
    structured array

    Parameters
    ----------
    nd: numpy.ndarray
        The array to view
    dtype: numpy.dtype or None (optional)
        The type of the structured array. If not provided, or None, nd.dtype is
        used.
    
    Returns
    -------
    A structured numpy.ndarray 
    """
    if nd.ndim not in (0, 1, 2):
        raise TypeError('np_nd_to_sa only takes 0, 1 or 2-dimensional arrays')

    if nd.ndim <= 1:
        nd = nd.reshape(nd.size, 1)
    if dtype is None:
        cols = nd.shape[1]
        dtype = np.dtype({'names' : map('f{}'.format, xrange(cols)), 
            'formats' : [nd.dtype for i in xrange(cols)]})

    return nd.reshape(nd.size).view(dtype)

def is_sa(A):
    """Returns true if the numpy.ndarray A is a structured array, false 
    otherwise."""
    return A.dtype.isbuiltin == 0

def np_type(val):
    """Returns a string or type that can be passed to numpy.dtype() to 
    generate the type of val"""
    if isinstance(val, str):
        return 'S{}'.format(len(val))
    return type(val)

def np_sa_to_dict(sa):
    """Converts an Numpy structured array with one row to a dictionary"""
    return {col_name : sa[col_name][0] for col_name in sa.dtype.names}

def dict_to_np_sa(d):
    """Converts a dict to a Numpy structured array with one row to a dict"""
    keys = d.keys()
    dtype = np.dtype([(key, np_type(d[key])) for key in keys])
    vals = [tuple([d[key] for key in keys])]
    return np.array(vals, dtype = dtype)
