from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase


class SQLRead(RunnableStage):
    """Stage to read in an sql table. Output is offered with the 'output' key

    Parameters
    ----------
    db_url : str
        The url of the database. Should conform to the format of
        SQLAlchemy database URLS
        (http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html#database-urls)
    table_name : str
        Name of the table which this UObject will represent
    conn_params : dict of str to ?
        A dictionary of the keyword arguments to be passed to the connect
        method of some library implementing the Python Database API
        Specification 2.0
        (https://www.python.org/dev/peps/pep-0249/#connect)

    """


    def __init__(self, db_url, table_name, conn_params={}):
        self.__db_url = db_url
        self.__table_name = table_name
        self.__conn_params = conn_params

    @property
    def input_keys(self):
        return []

    @property
    def output_keys(self):
        return ['output']

    def run(self, outputs_requested, **kwargs):
        uo = UObject(UObjectPhase.Write)
        uo.from_sql(
                self.__db_url, 
                self.__conn_params, 
                self.__table_name, 
                False)
        return {'output': uo}
