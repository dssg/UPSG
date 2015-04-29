from collections import namedtuple

from .uobject import UObjectException
from .stage import RunnableStage, MetaStage
from .pipeline import Pipeline
from .transform.identity import Identity
from .transform.split import SplitColumns, SplitColumn, Query
from .transform.timify import Timify
from .fetch.csv import CSVRead
from .fetch.sql import SQLRead
from .wrap.wrap_sklearn import wrap
from .model.multiclassify import Multiclassify

class ToasterError(Exception):
    pass

class DataToaster(MetaStage):
    """A convenient interface for building linear pipelines"""
    # TODO model with things other than multiclassify

    UNINITIALIZED, PREPROC, SPLIT, FINISHED = range(4)
    states = UNINITIALIZED, PREPROC, SPLIT, FINISHED

    class __Capper(RunnableStage):
        """captures output from the last stage of a pipeline and stores it 
        in-memory"""
        # TODO this isn't going to work if we're not running in debug mode

        def __init__(self, input_keys):
            self.__input_keys = input_keys
            self.result = dict.fromkeys(input_keys)

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return []

        def run(self, outputs_requested, **kwargs):
            for key in self.__input_keys:
                try: 
                    self.result[key] = kwargs[key].to_np()
                except UObjectException:
                    self.result[key] = kwargs[key].to_external_file()
            return {}

    def __init__(self):
        self.__pipeline = Pipeline()
        root_node = self.__pipeline.add(Identity(()))
        self.__input_node = root_node
        self.__output_node = root_node
        self.__state = self.UNINITIALIZED

    @property
    def input_keys(self):
        return self.__input_node.input_keys

    @property
    def output_keys(self):
        return self.__output_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__input_node, self.__output_node)

    def __node_id(self):
        return  self.__pipeline.add(Identity({'X_train_in': 'X_train',
                                              'y_train_in': 'y_train',
                                              'X_test_in': 'X_test',
                                              'y_test_in': 'y_test'}))

    def __from(self, stage_cls, *args, **kwargs):
        if self.__state != self.UNINITIALIZED:
            raise ToasterError('Data has already been imported.')
        # jettison the uninitialized pipeline
        self.__pipeline = Pipeline() 
        in_node = self.__pipeline.add(stage_cls(*args, **kwargs))
        self.__input_node = in_node
        self.__output_node = in_node
        self.__state = self.PREPROC
        return self

    def from_csv(self, file_name):
        self.__from(CSVRead, file_name)
        return self.__transform(Timify)

    def from_sql(self, db_url, table_name, conn_params={}):
        return self.__from(SQLRead, db_url, table_name, conn_params)

    FetchedConn = namedtuple('FetchedConn', ('conn', 'key'))

    def __latest_out_conn(self):
        return self.__get_out_conn(self.__output_node)

    def __get_out_conn(self, node):
        out_keys = node.output_keys
        possibilities = ('out', 'X_new', 'selected')
        for possibility in possibilities:
            if possibility in out_keys:
                return self.FetchedConn(node[possibility], possibility)
        raise ToasterError('Preproc stage doesn\'t have expected output key')

    def __get_in_conn(self, node):
        in_keys = node.input_keys
        possibilities = ('in', 'X_train')
        for possibility in possibilities:
            if possibility in in_keys:
                return self.FetchedConn(node[possibility], possibility)
        raise ToasterError('Preproc stage doesn\'t have expected input key')

    def __transform(self, stage_cls, *args, **kwargs):
        if self.__state == self.PREPROC:
            node = self.__pipeline.add(stage_cls(*args, **kwargs))
            in_conn = self.__get_in_conn(node).conn
            self.__latest_out_conn().conn > in_conn
            self.__output_node = node
            return self
        if self.__state == self.SPLIT:
            # If the stage is split, we apply the transform to both
            # TODO do we also want to transform y?
            node_train = self.__pipeline.add(stage_cls(*args, **kwargs))
            node_test = self.__pipeline.add(stage_cls(*args, **kwargs))
            node_id = self.__node_id()
            old_node_id = self.__output_node

            old_node_id['X_train'] > self.__get_in_conn(node_train).conn
            self.__get_out_conn(node_train).conn > node_id['X_train_in']

            old_node_id['X_test'] > self.__get_in_conn(node_test).conn
            self.__get_out_conn(node_test).conn > node_id['X_test_in']

            old_node_id['y_train'] > node_id['y_train_in']
            old_node_id['y_test'] > node_id['y_test_in']

            self.__output_node = node_id
            return self

        raise ToasterError('Not in preprocessing or split phase')

    def transform_with_sklearn(self, estimator, **kwargs):
        return self.__transform(wrap(estimator), **kwargs)

    def transform_select_cols(self, cols):
        return self.__transform(SplitColumns, cols)

    def __split(self, stage, y_col, train_out_key, test_out_key):
        if self.__state != self.PREPROC:
            raise ToasterError('Not in preprocessing phase')
        node_split_rows = self.__pipeline.add(stage)
        self.__latest_out_conn().conn > node_split_rows[node_split_rows.input_keys[0]]

        node_split_train = self.__pipeline.add(SplitColumn(y_col))
        node_split_rows[train_out_key] > node_split_train['in']

        node_split_test = self.__pipeline.add(SplitColumn(y_col))
        node_split_rows[test_out_key] > node_split_test['in']

        node_id = self.__node_id()
        node_split_train['X'] > node_id['X_train_in']
        node_split_train['y'] > node_id['y_train_in']
        node_split_test['X'] > node_id['X_test_in']
        node_split_test['y'] > node_id['y_test_in']

        self.__output_node = node_id
        self.__state = self.SPLIT
        return self

    def split_random(self, y_col, **kwargs):
        return self.__split(SplitTrainTest(n_arrays=1, **kwargs), y_col, 'train0', 'test0')

    def split_by_query(self, y_col, query):
        # The query selects which data is /training/
        return self.__split(Query(query), y_col, 'out', 'complement')

    def __model(self, stage_cls, *args, **kwargs):
        # TODO what about metrics phase?
        if self.__state != self.SPLIT:
            raise ToasterError('Not in split phase')
        node_model = self.__pipeline.add(stage_cls(*args, **kwargs))
        for key in ['X_train', 'X_test', 'y_train', 'y_test']:
            self.__output_node[key] > node_model[key]
        self.__output_node = node_model
        self.__state = self.FINISHED
        return self

    def classify_and_report(
            self, 
            report_file_name='report.html', 
            clf_and_params_dict=None,
            cv=2,
            metrics=None):
        return self.__model(Multiclassify,
            'score', 
            report_file_name, 
            clf_and_params_dict, 
            cv,
            metrics)

    def run(self, **kwargs):
        if self.__state == self.UNINITIALIZED:
            raise ToasterError('can\'t run uninitialized Toaster')
        outer_p = Pipeline()
        node_toaster = outer_p.add(self)
        if self.__state == self.PREPROC:
            latest_out_conn = self.__latest_out_conn()
            in_keys = (latest_out_conn.key,)
        else:
            in_keys = self.output_keys
        node_cap = outer_p.add(self.__Capper(in_keys))
        for key in in_keys:
            node_toaster[key] > node_cap[key]
        outer_p.run(**kwargs)
        return node_cap.get_stage().result

