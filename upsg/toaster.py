
class ToasterError(Exception):
    pass

class Toaster(MetaStage):
    """A convenient interface for building linear pipelines"""
    # TODO we really need to figure out how to do multi-stream. It will make
    # things make more sense


    UNINITIALIZED, PREPROC, SPLIT, FINISHED = range(3)
    States = UNINITIALIZED, PREPROC, SPLIT, FINISHED

    class __RootStage(RunnableStage):
        # A Nop Stage to put in uninitialized toasters
        @property
        def input_keys(self):
            return []

        @property
        def output_keys(self):
            return []

        def run(self, outputs_requested, **kwargs):
            return {}

    def __init__(self):
        self.__pipeline = Pipeline()
        root_node = self.__pipeline.add(self.__RootStage())
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
        return (self.__pipeline, self.__in_node, self.__out_node)

    def __latest_out_conn(self):
        raise NotImplementedError()

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
        self.__latest_out_conn > in_conn
        self.__output_node = node
        return self

    def transform_with_sklearn(self, estimator, **kwargs):
        return self.__transform(wrap_and_make_stage(estimator, **kwargs))

    def transform_select_features(self, columns):
        raise NotImplementedError()

    def __split(self, stage, y_col):
        if self.__state != self.PREPROC:
            raise ToasterError('Not in preprocessing phase')
        raise NotImplementedError()

    def split_random(self, y_col, **kwargs):
        return self.__split(SplitTrainTest(n_arrays=2, **kwargs), y_col)
        raise NotImplementedError()

    def split_by_query(self, y_col, query):
        raise NotImplementedError()

    def classify_and_report(
            self, 
            report_file_name='report.html', 
            clf_and_params_dict=None,
            metrics=None,
            do_run=True):
        raise NotImplementedError()

