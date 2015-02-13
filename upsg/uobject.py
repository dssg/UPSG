class UObjectPhase:
    """Enumeration of UObject phases

    UObjects are write-once. They must be written, then read.
    This enumeration specifies what is happening at present. 

    """
    Write, Read = range(2)

class UObject:
    """A universal object signifying intermediary state in a pipeline.

    Conceptually, this object is a write-once table. It can be written
    and read using a number of interfaces. For example, it can be treated
    as a table in a PostgreSQL database or a Pandas dataframe residing in
    memory. The internal representation is up to UPSG. Regardless of 
    internal representation, each UObject will be represented by a
    .upsg file that resides on the local disk. This .upsg file will be used
    to communicate between different steps in the pipeline.

    The interface to use will be chosen once when the UObject is being
    written and once when the UObject is being read. In order to choose
    an interface, first create a UObject instance, and then invoke one of
    its methods prefixed with "to_" to read or "from_" to write. 
    For example, to_postgres or from_dataframe. 

    If an object is invoked in write mode, it must be finalized
    before it can be read by another phase in the pipeline using one of
    the "to_" methods.

    """
    
    def __init__(self, phase, file_name = None):
        """initializer

        Prepares a UObject to be further used in a program. After a UObject
        instance is created, then the interface can be chosen and it can be
        read or written to in the rest of the program. Each instance of 
        UObject must be either read-only or write-only. 

        arguments:
        phase -- A member of UObjectPhase specifying whether the U_object
            is being written or read. Should be either UObjectPhase.Write
            or UObjectPhase.Read, respectively.
        file_name -- The name of the .upsg file representing this universal
            intermediary object. 

            If the file is being written, this argument is optional. If not 
            specified, an arbitrary, unique filename will be chosen. This
            filename can be found by invoking the get_file_name function.

            If the file is being read, this argument is mandatory. Failure
            to specify the argument will result in an exception.

        """
        #TODO stub
        pass

    def get_phase(self):
        """returns a member of UObjectPhase signifying whether the UObject
        is being read or written."""
        #TODO stub
        return UObjectPhase.Read
    
    def get_file_name(self):
        """Returns the path of this UObject's .upsg file."""
        #TODO stub
        return ''

    def is_finalized(self):
        """

        If the UObject is being written, returns a boolean signifying
        whether or not one of the "from_" methods has been called yet.

        If the UObject is being read, returns a boolean specifying
        whether or not one of the "to_" methods has been called yet.

        """
        #TODO stub
        return False
    
    def to_postgresql(self): 
        """Makes the universal object available in postgres.

        returns a tuple (connection_string, table_name)

        """
        #TODO stub
        return ('', '')
    
    def to_dataframe(self):
        """Makes the universal object available in a Python dataframe.

        returns a Pandas dataframe containing a representation of the object.

        """
        #TODO stub
        return None
    
    def to_tuple_of_dicts(self):
        """Makes the universal object available in a tuple of dictionaries.

        returns a tuple of dictionaries containing a representation of the
        object.

        This is probably the choice to use when a universal object encodes
        parameters for a model.
        
        """
        #TODO stub
        return ()

    def from_postgres(self, con_string, query):
        """Writes the results of a query to the universal object and prepares
        the .upsg file.

        """
        #TODO stub
        pass

    def from_dataframe(self, dataframe):
        """Writes contents of a Pandas dataframe to the universal object and 
        prepares the .upsg file.

        """
        #TODO stub
        pass

    def from_tuple_of_dicts(self, tuple_of_dicts):
        """Writes contents of a tuple of dictionaries to the universal object
        and prepares the .upsg file.

        This is probably the choice to use when a universal object encodes
        parameters for a model.
        """
        #todo stub
        pass
