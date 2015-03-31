import numpy as np
import itertools as it
import re
import uuid
from datetime import datetime
from sqlalchemy.schema import Table, Column
from sqlalchemy import MetaData
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
import sqlalchemy.types as sqlt


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
    # TODO use a better metric to determine the datatype
    view = sa.view(dtype=sa.dtype[0])
    if len(sa.dtype) == 1:
        if sa.size == 1:
            return (view.reshape(()), sa.dtype)
        return (view.reshape(len(sa)), sa.dtype)
    return (view.reshape(len(sa), -1), sa.dtype)


def np_nd_to_sa(nd, dtype=None):
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
        dtype = np.dtype({'names': map('f{}'.format, xrange(cols)),
                          'formats': [nd.dtype for i in xrange(cols)]})

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
    if '_UPSG_EMPTY_DICT' in sa.dtype.names:
        # because Numpy doesn't let us get away with 0-column arrays
        return {}
    return {col_name: sa[col_name][0] for col_name in sa.dtype.names}


def dict_to_np_sa(d):
    """Converts a dict to a Numpy structured array with one row to a dict"""
    if not d:
        # because Numpy doesn't let us get away with 0-column arrays
        return np.array([], dtype=[('_UPSG_EMPTY_DICT', 'S1')])
    keys = d.keys()
    dtype = np.dtype([(key, np_type(d[key])) for key in keys])
    vals = [tuple([d[key] for key in keys])]
    return np.array(vals, dtype=dtype)


# http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
re_utf_to_ascii = re.compile(r'[^\x00-\x7F]+')


def utf_to_ascii(s):
    """Converts a unicode string to an ascii string
    
    If the passed object is a unicode string, replaces unicode characters
    with '.' Each contiguous sequence of characters will be replaced with a 
    single '.' Consequently, the length of the replaced string will not exceed
    the length of the given string

    If the passed object is not a unicode string, returns the object

    """

    if isinstance(s, unicode):
        return str(re_utf_to_ascii.sub('.', s))
    return s

# TODO I'm missing a lot of these, most notably datetime, which we don't have
# natively so we have to do some fancy conversion
np_to_sql_types = {
    np.dtype(bool): (sqlt.BOOLEAN,),
    np.dtype(int): (sqlt.INTEGER, sqlt.BIGINT, sqlt.SMALLINT),
    np.dtype(float): (sqlt.FLOAT, sqlt.DECIMAL, sqlt.REAL, sqlt.NUMERIC),
    np.dtype(str): (sqlt.VARCHAR, sqlt.CHAR, sqlt.NCHAR, sqlt.NVARCHAR,
                    sqlt.TEXT),
    np.dtype('datetime64[s]'): (sqlt.DATETIME, sqlt.DATE, sqlt.TIME,
                                sqlt.TIMESTAMP)
}
# TODO other time resolutions. (But we do have to specify 1 of them)
# http://stackoverflow.com/questions/16618499/numpy-datetime64-in-recarray
# I believe ns is the internal one:
# https://github.com/pydata/pandas/issues/6741
sql_to_np_types = {
    sql_type: np_type for sql_type,
    np_type in it.chain.from_iterable(
        (it.izip(
            np_to_sql_types[npt],
            it.repeat(npt)) for npt in np_to_sql_types))}


def sql_to_np(tbl, conn):
    """Converts a sql table to a Numpy structured array.

    Parameters
    ----------
    tbl : sqlalchemy.schema.table
        Table to convert
    conn : sqlalchemy.engine.Connectable
        Connection to use to connect to the database

    Returns
    -------
    A Numpy structured array

    """

    # todo sessionmaker is somehow supposed to be global
    Session = sessionmaker(bind=conn)
    session = Session()
    # first pass, we don't worry about string length
    dtype = [(str(col.name), sql_to_np_types[type(col.type)]) for
             col in tbl.columns]
    # now, we find the max string length for our char columns
    str_cols = [tbl.columns[col_name] for col_name, col_dtype in dtype if
                col_dtype == np.dtype(str)]
    query_funcs = [func.max(func.length(col)).label(col.name) for
                   col in str_cols]
    query = session.query(*query_funcs)
    str_lens = {col_name: str_len for col_name, str_len in it.izip(
        (desc['name'] for desc in query.column_descriptions),
        query.one())}

    def corrected_col_dtype(name, col_dtype):
        if col_dtype == np.dtype(str):
            return (name, '|S{}'.format(str_lens[name]))
        return (name, col_dtype)
    dtype_corrected = np.dtype([corrected_col_dtype(*dtype_tuple) for
                                dtype_tuple in dtype])
    # np.fromiter can't directly use the results of a query:
    #   http://mail.scipy.org/pipermail/numpy-discussion/2010-August/052358.html
    # TODO deal with unicode (which numpy can't handle)
    return np.fromiter((tuple(row) for row in
                        session.query(tbl).all()), dtype=dtype_corrected)

# http://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
NP_EPOCH = np.datetime64('1970-01-01T00:00:00Z')
NP_SEC_DELTA = np.timedelta64(1, 's')


def datetime64_to_datetime(dt):
    """Converts a numpy.datatime64 to a Python datetime

    If the argument is not a numpy.datetime64, returns the argument

    """
    if not isinstance(dt, np.datetime64):
        return dt
    return datetime.utcfromtimestamp((dt - NP_EPOCH) / NP_SEC_DELTA)


def np_to_sql(A, tbl_name, conn):
    """Converts a numpy structured array to an sql table

    Parameters
    ----------
    A : numpy structured array
        Array to convert

    tbl_name : str
        Name of table to insert into sql database

    conn : sqlalchemy.engine.Connectable
        Connection for the sql database

    Returns
    -------
    sqlalchemy.schema.table corresponding to the uploaded structured array

    """
    dtype = A.dtype
    col_names = dtype.names

    def sql_dtype(col_dtype):
        if col_dtype.char == 'S':
            return sqlt.VARCHAR(col_dtype.itemsize)
        return np_to_sql_types[col_dtype][0]
    cols = [Column(name, sql_dtype(dtype[name])) for name in col_names]
    md = MetaData()
    tbl = Table(tbl_name, md, *cols)
    md.create_all(conn)
    # http://stackoverflow.com/questions/7043158/insert-numpy-array-into-mysql-database
    # TODO find a faster way to fix datetimes
    conn.execute(
        tbl.insert(), [
            dict(
                it.izip(
                    col_names, [
                        datetime64_to_datetime(cell) for cell in row])) for 
                                row in A])
    return tbl


def random_table_name():
    """Returns a random table name prefixed with _UPSG_ that is unlikely
    to collide with another random table name"""
    return ('_UPSG_' + str(uuid.uuid4())).replace('-', '_')
