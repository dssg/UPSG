import os 
import inspect
import itertools as it
import re
import uuid
import upsg
import cgi
import importlib
from datetime import datetime
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from sqlalchemy.schema import Table, Column
from sqlalchemy import MetaData
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
import sqlalchemy.types as sqlt

UPSG_PATH = os.path.dirname(inspect.getfile(upsg))
REPO_PATH = os.path.join(UPSG_PATH, '..')
RESOURCES_PATH = os.path.join(UPSG_PATH, 'resources')

def get_resource_path(file_name):
    """given the name of a resource, returns the full path"""
    return os.path.join(RESOURCES_PATH, file_name)

__type_permissiveness_ranks = {'b': 0, 'M': 100, 'i': 200, 'f': 300, 'S': 400}
def __type_permissiveness(dtype):
    # TODO handle other types
    return __type_permissiveness_ranks[dtype.kind] + dtype.itemsize

def np_dtype_is_homogeneous(A):
    """True iff dtype is nonstructured or every sub dtype is the same"""
    # http://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
    if not is_sa(A):
        return True
    dtype = A.dtype
    first_dtype = dtype[0]
    return all(dtype[i] == first_dtype for i in xrange(len(dtype)))

def np_sa_to_nd(sa):
    """
    
    Returns a view of a numpy structured array as a single-type 1 or
    2-dimensional array. If the resulting nd array would be a column vector,
    returns a 1-d array instead. If the resulting array would have a single
    entry, returns a 0-d array instead

    All elements are converted to the most permissive type. permissiveness
    is determined first by finding the most permissive type in the ordering:
    datetime64 < int < float < string
    then by selecting the longest typelength among all columns with with that
    type.

    If the sa does not have a homogeneous datatype already, this may require
    copying and type conversion rather than just casting. Consequently, this
    operation should be avoided for heterogeneous arrays

    Based on http://wiki.scipy.org/Cookbook/Recarray.

    Parameters
    ----------
    sa : numpy.ndarray
        The structured array to view

    Returns
    -------
    tuple (nd, dtype)
        where nd is numpy.ndarray array view and dtype is the numpy.dtype of
        the structured array that was passed in.
    """
    dtype = sa.dtype
    if len(dtype) == 1:
        if sa.size == 1:
            return (sa.view(dtype=dtype[0]).reshape(()), dtype)
        return (sa.view(dtype=dtype[0]).reshape(len(sa)), dtype)
    if np_dtype_is_homogeneous(sa):
        return (sa.view(dtype=dtype[0]).reshape(len(sa), -1), dtype)
    # If type isn't homogeneous, we have to convert
    dtype_it = (dtype[i] for i in xrange(len(dtype)))
    most_permissive = max(dtype_it, key=__type_permissiveness)
    col_names = dtype.names
    cols = (sa[col_name].astype(most_permissive) for col_name in col_names)
    nd = np.column_stack(cols)
    return (nd, dtype)

def np_nd_to_sa(nd, dtype=None):
    """
    
    Returns a view of a numpy, single-type, 0, 1 or 2-dimensional array as a
    structured array

    Parameters
    ----------
    nd : numpy.ndarray
        The array to view
    dtype : numpy.dtype or None (optional)
        The type of the structured array. If not provided, or None, nd.dtype is
        used for all columns.

        If the dtype requested is not homogeneous and the datatype of each
        column is not identical nd.dtype, this operation may involve copying
        and conversion. Consequently, this operation should be avoided with
        heterogeneous or different datatypes.

    Returns
    -------
    A structured numpy.ndarray
    """
    if is_sa(nd):
        return nd
    if nd.ndim not in (0, 1, 2):
        raise TypeError('np_nd_to_sa only takes 0, 1 or 2-dimensional arrays')
    nd_dtype = nd.dtype
    if nd.ndim <= 1:
        nd = nd.reshape(nd.size, 1)
    if dtype is None:
        n_cols = nd.shape[1]
        dtype = np.dtype({'names': map('f{}'.format, xrange(n_cols)),
                          'formats': [nd_dtype for i in xrange(n_cols)]})
        return nd.reshape(nd.size).view(dtype)
    type_len = nd_dtype.itemsize
    if all(dtype[i] == nd_dtype for i in xrange(len(dtype))):
        return nd.reshape(nd.size).view(dtype)
    # if the user requests an incompatible type, we have to convert
    cols = (nd[:,i].astype(dtype[i]) for i in xrange(len(dtype))) 
    return np.array(it.izip(*cols), dtype=dtype)


def is_sa(A):
    """
    
    Returns true if the numpy.ndarray A is a structured array, false
    otherwise.
    
    """
    return A.dtype.isbuiltin == 0


def np_type(val):
    """
    
    Returns a string or type that can be passed to numpy.dtype() to
    generate the type of val
    
    """
    if isinstance(val, basestring):
        return 'S{}'.format(len(val))
    return type(val)


def __fix_dict_typing(o):
    if isinstance(o, np.string_):
        if o == '_UPSG_NONE_':
            return None
        return str(o)
    return o

def np_sa_to_dict(sa):
    """Converts an Numpy structured array with one row to a dictionary"""
    if '_UPSG_EMPTY_DICT' in sa.dtype.names:
        # because Numpy doesn't let us get away with 0-column arrays
        return {}
    return {col_name: __fix_dict_typing(sa[col_name][0]) for 
            col_name in sa.dtype.names}
    #return {col_name: sa[col_name][0] for col_name in sa.dtype.names}


def dict_to_np_sa(d):
    """Converts a dict to a Numpy structured array with one row to a dict"""
    if not d:
        # because Numpy doesn't let us get away with 0-column arrays
        return np.array([], dtype=[('_UPSG_EMPTY_DICT', 'S1')])
    # because Numpy/pytables doesn't seem to have a good way to encode None
    # TODO a better way to encode None?
    keys = d.keys()
    d = {key: '_UPSG_NONE_' if d[key] is None else d[key] for 
                  key in d} 
    dtype = np.dtype([(utf_to_ascii(key), np_type(d[key])) for key in keys])
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
        return s.encode('ascii', 'replace')
    return s

# TODO I'm missing a lot of these, most notably datetime, which we don't have
# natively so we have to do some fancy conversion
np_to_sql_types = {
    np.dtype(bool): (sqlt.BOOLEAN, sqlt.Boolean),
    np.dtype(int): (sqlt.INTEGER, sqlt.BIGINT, sqlt.SMALLINT, sqlt.BigInteger,
                    sqlt.Integer, sqlt.SmallInteger),
    np.dtype(float): (sqlt.FLOAT, sqlt.DECIMAL, sqlt.REAL, sqlt.NUMERIC, 
                      sqlt.Float, sqlt.Numeric),
    np.dtype(str): (sqlt.VARCHAR, sqlt.CHAR, sqlt.NCHAR, sqlt.NVARCHAR,
                    sqlt.TEXT, sqlt.String, sqlt.Text, sqlt.Unicode,
                    sqlt.UnicodeText),
    np.dtype('datetime64[s]'): (sqlt.DATETIME, sqlt.DATE, sqlt.TIME,
                                sqlt.TIMESTAMP, sqlt.Date, sqlt.DateTime,
                                sqlt.Time)
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
    dtype = []
    for col in tbl.columns:
        sql_type = col.type
        np_type = None
        try:
            np_type = sql_to_np_types[type(sql_type)]
        except KeyError:
            for base_sql_type in sql_to_np_types:
                if isinstance(sql_type, base_sql_type):
                #if base_sql_type in inspect.getmro(type(sql_type)):
                    np_type = sql_to_np_types[base_sql_type]
                    continue
        # TODO nice error if we still don't find anything
        if np_type is None:
            raise KeyError('Type not found ' + str(sql_type))
            # TODO a more appropriate type of error
        dtype.append((str(col.name), np_type))
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
    return np.fromiter((np_process_row(row, dtype_corrected) for row in
                        session.query(tbl).all()), dtype=dtype_corrected)


def np_process_row_elmt(entry, dtype):
    if entry is None:
        if 'S' in dtype:
            return ''
        if 'i' in dtype or 'l' in dtype:
            return -999
            # TODO is there some better way to handle null ints
        return np.nan
    if 'S' in dtype:
        return utf_to_ascii(entry)
    return entry
    

def np_process_row(row, dtype):
    return tuple([np_process_row_elmt(entry, dtype[idx].str) for idx, entry in
                  enumerate(row)])

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
    sqlalchemy.schema.table 
        sqlalchemy table corresponding to the uploaded structured array

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
    """
    
    Returns a random table name prefixed with _UPSG_ that is unlikely
    to collide with another random table name
    
    """
    return ('_UPSG_' + str(uuid.uuid4())).replace('-', '_')


def html_escape(s):
    """Returns a string with all its html-averse characters html escaped"""
    return cgi.escape(s).encode('ascii', 'xmlcharrefreplace')

def html_format(fmt, *args, **kwargs):
    clean_args = [html_escape(str(arg)) for arg in args]
    clean_kwargs = {key: html_escape(str(kwargs[key])) for 
                    key in kwargs}
    return fmt.format(*clean_args, **clean_kwargs)

def import_object_by_name(target):
    """Imports an object given its fully qualified package name

    Examples
    --------

    >>> SVC = import_by_name('sklearn.svm.svc')

    """

    split = target.split('.')
    object_name = split[-1]
    module_name = '.'.join(split[:-1])
    skl_module = importlib.import_module(module_name)
    return skl_module.__dict__[object_name]

def obj_to_str(sa):
    """
    
    Takes a structured array with columns of "O" (object) dtype and converts 
    them to columns of "S" (string) type. The length of the string columns are
    the length of the longest string in the column

    """
    cols = []
    ndtype = []
    for col_name, sub_dtype in sa.dtype.descr:
        col = sa[col_name]
        if 'O' in sub_dtype:
            # TODO our assumption that these are strings is not really 
            # justified
            field_len = len(max(col, key=len))
            nsub_dtype = 'S{}'.format(field_len)
            cols.append(col.astype(nsub_dtype))
            ndtype.append((col_name, nsub_dtype))
            continue
        cols.append(col)
        ndtype.append((col_name, sub_dtype))
    return merge_arrays(cols).view(dtype=ndtype)

def np_to_html_table(sa, fout):
    fout.write('<p>table of shape: ({},{})</p>'.format(
        len(sa),
        len(sa.dtype)))
    fout.write('<p><table>\n')
    header = '<tr>{}</tr>\n'.format(
        ''.join(
                [html_format(
                    '<th>{}</th>',
                    name) for 
                 name in sa.dtype.names]))
    fout.write(header)
    rows = sa[:100]
    data = '\n'.join(
        ['<tr>{}</tr>'.format(
            ''.join(
                [html_format(
                    '<td>{}</td>',
                    cell) for
                 cell in row])) for
         row in rows])
    fout.write(data)
    fout.write('\n')
    fout.write('</table></p>')


