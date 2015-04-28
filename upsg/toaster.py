
class ToasterError(Exception):
    pass

class Toaster(MetaStage):
    """A convenient interface for building linear pipelines"""

    UNINITIALIZED, PREPROC, SPLIT, FINISHED = range(3)
    States = UNINITIALIZED, PREPROC, SPLIT, FINISHED

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

    def __latest_out_conn(self):
        out_node = self.__output_node
        out_keys = out_node.output_keys
        possibilities = ('out', 'X_new', 'selected')
        for possibility in possibilities:
            if possibility in out_keys:
                return out_node[possibility]
        raise ToasterError('Preproc stage doesn\'t have expected output key')

    def __from(self, stage):
        if self.__state != self.UNINITIALIZED:
            raise ToasterError('Data has already been imported.')
        # jettison the uninitialized pipeline
        self.__pipeline = Pipeline() 
        in_node = self.__pipeline.add(stage)
        self.__input_node = in_node
        self.__output_node = in_node
        self.__stage = self.PREPROC
        return self

    def from_csv(self, file_name):
        return self.__from(self, CSVRead(file_name))

    def from_sql(self, db_url, table_name, conn_params={}):
        return self.__from(self, SQLRead(db_url, table_name, conn_params))

    def __transform(self, stage)
        if self.__state != self.PREPROC:
            raise ToasterError('Not in preprocessing phase')
        node = self.__pipeline.add(stage)
        try: 
            in_conn = stage['in']
        except KeyError:
            try:
                in_conn = stage['X_train']
            except KeyError:
                raise ToasterError(('Not a valid transformation. Doesn\'t'
                                    'support \'in\' or \'X_train\' keys'))
        self.__latest_out_conn() > in_conn
        self.__output_node = node
        return self

    def transform_with_sklearn(self, estimator, **kwargs):
        return self.__transform(wrap_and_make_stage(estimator, **kwargs))

    def transform_select_cols(self, cols):
        return self.__transform(SplitColumns(cols))

    def __split(self, stage, y_col, train_out_key, test_out_key):
        if self.__state != self.PREPROC:
            raise ToasterError('Not in preprocessing phase')
        node_split_rows = self.__pipeline.add(stage)
        self.__latest_out_conn() > node_split_rows[node_split_rows.input_keys[0]]

        node_split_train = p.add(SplitColumn(y_col))
        node_split_rows[train_out_key] > node_split_train['in']

        node_split_test = p.add(SplitColumn(y_col))
        node_split_rows[test_out_key] > node_split_test['in']

        node_id = self.__pipeline.add(Identity({'X_train_in': 'X_train',
                                                'y_train_in': 'y_train',
                                                'X_test_in': 'X_test',
                                                'y_test_in': 'y_test'}))
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

    def __model(self, stage):
        if self.__state != self.SPLIT:
            raise ToasterError('Not in split phase')
        node_model = self.__pipeline.add(stage)
        for key in ['X_train', 'X_test', 'y_train', 'y_test']:
            self.__output_node[key] > node_model[key]
        self.__output_node = node_model
        self.__state = self.FINISHED

    def run(self):
        self.__pipeline.run()

    def classify_and_report(
            self, 
            report_file_name='report.html', 
            clf_and_params_dict=None,
            cv=2,
            metrics=None,
            do_run=True):
        ret = self.__model(Multiclassify(
            'score', 
            report_file_name, 
            clf_and_params_dict, 
            cv,
            metrics))
        if do_run:
            self.run()
        return ret
