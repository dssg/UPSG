================
The .upsg Format
================
In principal, any program that writes and reads .upsg files can participate
in a UPSG pipeline. The format specification follows.

1. A .upsg file is an `HDF5 <https://www.hdfgroup.org/HDF5/>`_ file
2. A .upsg file conforms to the format used by 
   `PyTables <http://www.pytables.org/usersguide/file_format.html>`_.
3. Every .upsg file has a group "/upsg_inf" with an attribute called 
   "storage_method". The value of storage_method determines how the rest
   of the file is stored.
   
    *np*   
        If the storage_method is "np" then the table is stored internally as
        a table created using the PyTables 
        `create_table <http://www.pytables.org/usersguide/libref/file_class.html?highlight=file#tables.File.create_table>`_
        method. This table will be called "table" and will be located in the
        "/np" group. 

        If the table had a column of type 
        `datetime64 <http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html>`_, 
        the column will appear as an int64 inside the table. There will be an 
        additional table in the /np group called "dt_cols". The first column 
        of dt_cols is the name of the column that should be interpreted as a 
        datetime64. The second column of dt_cols is a string representation 
        of the particular flavor of datetime64 which the column should be
        interpreted as. For example "<M8[s]" or "<M8[ms]".

        If the table is meant to be be interpreted as a dictionary, then it
        will have one row. The name of each column is the key, and the entry
        in row 0 of that column is the value. If the dictionary was supposed
        to be empty, there will be a single column with the name 
        _UPSG_EMPTY_DICT

    *sql*
        If the storage method is "sql", there will be a gropu called "/sql".
        That group will have the following attributes:

        db_url
            The 
            `SQLAlchemy url <http://docs.sqlalchemy.org/en/latest/core/engines.html>`_
            used to connect to the database
        tbl_name
            The name of the table in the database that this file represents
        conn_params
            A table that encodes a dictionary as in the "np" storage 
            specification. The dictionary encodes connection parameters used
            in the `DB API 2 <https://www.python.org/dev/peps/pep-0249/>`_
            connect method.

