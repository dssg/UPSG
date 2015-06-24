import ast
import numpy as np
from os import system
import unittest

from numpy.lib.recfunctions import append_fields
from numpy.lib.recfunctions import merge_arrays

import pandas as pd

from sklearn.cross_validation import KFold as SKKFold

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.export.np import NumpyWrite
from upsg.fetch.csv import CSVRead
from upsg.fetch.np import NumpyRead
from upsg.transform.rename_cols import RenameCols
from upsg.transform.sql import RunSQL
from upsg.transform.split import Query, SplitColumns, KFold, SplitByInds
from upsg.transform.fill_na import FillNA
from upsg.transform.label_encode import LabelEncode
from upsg.transform.lambda_stage import LambdaStage
from upsg.transform.timify import Timify
from upsg.transform.identity import Identity
from upsg.transform.apply_to_selected_cols import ApplyToSelectedCols
from upsg.transform.merge import Merge
from upsg.transform.hstack import HStack
from upsg.wrap.wrap_sklearn import wrap
from upsg.utils import np_nd_to_sa, np_sa_to_nd, is_sa, obj_to_str

from utils import path_of_data, UPSGTestCase, csv_read


class TestTransform(UPSGTestCase):

    def test_rename_cols(self):
        infile_name = path_of_data('mixed_csv.csv')
        rename_dict = {'name': 'designation', 'height': 'tallness'}

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        trans_node = p.add(RenameCols(rename_dict))
        csv_write_node = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_read_node['output'] > trans_node['input']
        trans_node['output'] > csv_write_node['input']

        self.run_pipeline(p)

        control = {'id', 'designation', 'tallness'}
        result = set(self._tmp_files.csv_read('out.csv').dtype.names)

        self.assertTrue(np.array_equal(result, control))

    def test_sql(self):

        # Make sure we don't accidentally corrupt our test database
        db_path, db_file_name = self._tmp_files.tmp_copy(path_of_data(
            'small.db'))
        db_url = 'sqlite:///{}'.format(db_path)
        
        q_sel_employees = 'CREATE TABLE {tmp_emp} AS SELECT * FROM employees;'
        # We have to be careful about the datetime type in sqlite3. It will
        # forget if we don't keep reminding it, and if it forgets sqlalchemy
        # will be unhappy. Hence, we can't use CREATE TABLE AS if our table
        # has a DATETIME
        q_sel_hours = ('CREATE TABLE {tmp_hrs} '
                       '(id INT, employee_id INT, time DATETIME, '
                       '    event_type TEXT); '
                       'INSERT INTO {tmp_hrs} SELECT * FROM hours;')
        q_join = ('CREATE TABLE {joined} '
                  '(id INT, last_name TEXT, salary REAL, time DATETIME, '
                  '    event_type TEXT); '
                  'INSERT INTO {joined} '
                  'SELECT {tmp_emp}.id, last_name, salary, time, event_type '
                  'FROM {tmp_emp} JOIN {tmp_hrs} ON '
                  '{tmp_emp}.id = {tmp_hrs}.employee_id;')

        p = Pipeline()
        get_emp = p.add(RunSQL(db_url, q_sel_employees, [], ['tmp_emp'], {}))
        get_hrs = p.add(RunSQL(db_url, q_sel_hours, [], ['tmp_hrs'], {}))
        join = p.add(RunSQL(db_url, q_join, ['tmp_emp', 'tmp_hrs'], ['joined'],
                            {}))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        get_emp['tmp_emp'] > join['tmp_emp']
        get_hrs['tmp_hrs'] > join['tmp_hrs']
        join['joined'] > csv_out['input']

        self.run_pipeline(p)

        ctrl = csv_read(path_of_data('test_transform_test_sql_ctrl.csv'))
        result = self._tmp_files.csv_read('out.csv')
        # Because Numpy insists on printing times with local offsets, but
        # not every computer has the same offset, we have to force it back
        # into UTC
        for i, dt in enumerate(result['time']):
            # .item() makes a datetime, which we can format correctly later
            # http://stackoverflow.com/questions/25134639/how-to-force-python-print-numpy-datetime64-with-specified-timezone
            result['time'][i] = np.datetime64(dt).item().strftime(
                    '%Y-%m-%dT%H:%M:%S')
        # Then we have to make the string field smaller
        new_cols = []
        for col in result.dtype.names:
            new_cols.append(result[col].astype(ctrl.dtype[col]))
        result = merge_arrays(new_cols, flatten=True) 
        result.dtype.names = ctrl.dtype.names

        self.assertTrue(np.array_equal(result, ctrl))

    def test_split_columns(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('numbers.csv')))
        split = p.add(SplitColumns(('F1', 'F3')))
        csv_out_sel = p.add(CSVWrite(self._tmp_files('out_sel.csv')))
        csv_out_rest = p.add(CSVWrite(self._tmp_files('out_rest.csv')))

        csv_in['output'] > split['input']
        split['output'] > csv_out_sel['input']
        split['complement'] > csv_out_rest['input']

        self.run_pipeline(p)
        
        result = self._tmp_files.csv_read('out_sel.csv')
        ctrl = csv_read(path_of_data('test_split_columns_ctrl_selected.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

        result = self._tmp_files.csv_read('out_rest.csv')
        ctrl = csv_read(path_of_data('test_split_columns_ctrl_rest.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

    def __test_ast_trans(self, raw, target, col_names):    
        # There doesn't seem to be a better easy way to test AST equality
        # than seeing if their dumps are equal:
        # http://stackoverflow.com/questions/3312989/elegant-way-to-test-python-asts-for-equality-not-reference-or-object-identity
        # If it gets to be a problem, we'll hand-roll something
        q = Query(raw)
        ctrl = ast.dump(ast.parse(target, mode='eval'))
        result = q.dump_ast(col_names)
        self.assertEqual(result, ctrl)

    def test_query_ast(self):
        # Make sure we can support simple queries
        ast_tests = [("id < 10", 
                      "in_table['id'] < 10", 
                      ('id', 'name')),
                     ("name == 'Bruce'", 
                      "in_table['name'] == 'Bruce'",
                      ('id', 'name')),
                     ("(id < 10) or (name == 'Bruce' and hired_dt != stop_dt)",
                      ("np.logical_or("
                           "in_table['id'] < 10, "
                           "np.logical_and("
                               "in_table['name'] == 'Bruce', "
                               "in_table['hired_dt'] != in_table['stop_dt']))"),
                      ('id', 'name', 'hired_dt', 'stop_dt')),
                     ("id >= 5 and not (terminated or not salary < 10000)",
                      ("np.logical_and("
                           "in_table['id'] >= 5, "
                           "np.logical_not("
                               "np.logical_or("
                                   "in_table['terminated'], "
                                   "np.logical_not("
                                        "in_table['salary'] < 10000))))"),
                      ('id', 'terminated', 'salary', 'demerits')),
                     ("start_date < DT('2012-04-19')",
                      "in_table['start_date'] < np.datetime64('2012-04-19')",
                      ('start_date',))]
        for raw, target, col_names in ast_tests:
            self.__test_ast_trans(raw, target, col_names)

    def test_query_complex(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('query.csv')))
        q1_node = p.add(Query("((id == value) and not (use_this_col == 'no'))"
                              "or name == 'fish'"))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))
        csv_comp = p.add(CSVWrite(self._tmp_files('out_comp.csv')))

        csv_in['output'] > q1_node['input']
        q1_node['output'] > csv_out['input']
        q1_node['complement'] > csv_comp['input']

        self.run_pipeline(p)

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('query_ctrl.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

        result = self._tmp_files.csv_read('out_comp.csv')
        ctrl = csv_read(path_of_data('query_ctrl_comp.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

    def test_query_dates(self):

        p = Pipeline()

        dates = np.array([(np.datetime64('2012-01-01')), 
                          (np.datetime64('2013-04-05')), 
                          (np.datetime64('2014-03-11')),
                          (np.datetime64('2015-01-01'))], dtype=[('dt', 'M8[D]')])
        
        inds = np.array([(i,) for i in xrange(dates.size)], dtype=[('f0', int)])

        np_in = p.add(NumpyRead(dates))

        q2_node = p.add(Query("dt <= DT('2014-01-01')"))
        np_in['output'] > q2_node['input']

        np_out = p.add(NumpyWrite())
        q2_node['output'] > np_out['input']

        np_complement = p.add(NumpyWrite())
        q2_node['complement'] > np_complement['input']

        np_out_inds = p.add(NumpyWrite())
        q2_node['output_inds'] > np_out_inds['input']

        np_complement_inds = p.add(NumpyWrite())
        q2_node['complement_inds'] > np_complement_inds['input']

        self.run_pipeline(p)

        self.assertTrue(np.array_equal(np_out.get_stage().result, dates[:2]))
        self.assertTrue(np.array_equal(np_complement.get_stage().result, dates[2:]))
        self.assertTrue(np.array_equal(np_out_inds.get_stage().result, inds[:2]))
        self.assertTrue(np.array_equal(np_complement_inds.get_stage().result, inds[2:]))

    def test_fill_na(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('missing_vals_mixed.csv')))
        fill_na = p.add(FillNA(-1))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_in['output'] > fill_na['input']
        fill_na['output'] > csv_out['input']

        self.run_pipeline(p)

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('test_transform_test_fill_na_ctrl.csv'))
        
        self.assertTrue(np.array_equal(result, ctrl))

    def test_label_encode(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('categories.csv')))
        le = p.add(LabelEncode())
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_in['output'] > le['input']
        le['output'] > csv_out['input']

        self.run_pipeline(p)

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('test_transform_test_label_encode_ctrl.csv'))
        
        self.assertTrue(np.array_equal(result, ctrl))

    def test_kfold(self):

        folds = 3
        rows = 6

        X = np.random.randint(0, 1000, (rows, 3))
        y = np.random.randint(0, 1000, (rows, 1))

        p = Pipeline()

        np_in_X = p.add(NumpyRead(X))
        np_in_y = p.add(NumpyRead(y))

        kfold = p.add(KFold(2, folds, random_state=0))
        np_in_X['output'] > kfold['input0']
        np_in_y['output'] > kfold['input1']

        ctrl_kf = SKKFold(rows, n_folds = folds, random_state=0)
        out_files = []
        expected_folds = []
        arrays = (X, y)
        for fold_i, train_test_inds in enumerate(ctrl_kf):
            for array_i, array in enumerate(arrays):
                for select_i, selection in enumerate(('train', 'test')):
                    out_key = '{}{}_{}'.format(selection, array_i, fold_i) 
                    out_file = out_key + '.csv'
                    out_files.append(out_file)
                    stage = p.add(CSVWrite(self._tmp_files(out_file)))
                    kfold[out_key] > stage['input']
                    slice_inds = train_test_inds[select_i]
                    expected_folds.append(
                            np_nd_to_sa(arrays[array_i][slice_inds]))

        self.run_pipeline(p)

        for out_file, expected_fold in zip(out_files, expected_folds):
            self.assertTrue(np.array_equal(
                self._tmp_files.csv_read(out_file),
                expected_fold))

    def test_lambda(self):

        # Test output key generation

        l1 = LambdaStage(lambda x, y: 0)
        self.assertEqual(l1.input_keys, ['x', 'y'])
        self.assertEqual(l1.output_keys, ['output0',])

        l2 = LambdaStage(lambda: 0, n_outputs=3)
        self.assertEqual(l2.input_keys, [])
        self.assertEqual(l2.output_keys, ['output{}'.format(i) for i in
                                          xrange(3)])

        # Test running in pipeline

        in_data = np_nd_to_sa(np.random.random((100, 10)))
        scale = np_nd_to_sa(np.array(3))
        out_keys = ['augmented', 'log_col', 'sqrt_col', 'scale_col'] 

        def log1_sqrt2_scale3(A, scale):
            names = A.dtype.names
            log_col = np.log(A[names[0]])
            sqrt_col = np.sqrt(A[names[1]])
            scale_col = A[names[2]] * scale[0][0]

            return (append_fields(
                        A, 
                        ['log1', 'sqrt2', 'scale3'], 
                        (log_col, sqrt_col, scale_col)),
                    log_col,
                    sqrt_col,
                    scale_col)

        p = Pipeline()

        np_in = p.add(NumpyRead(in_data))
        scale_in = p.add(NumpyRead(scale))

        lambda_stage = p.add(
            LambdaStage(
                log1_sqrt2_scale3, 
                out_keys))
        np_in['output'] > lambda_stage['A']
        scale_in['output'] > lambda_stage['scale']

        csv_out_stages = []
        for key in out_keys:
            stage = p.add(
                    CSVWrite(
                        self._tmp_files(
                            'out_{}.csv'.format(key))))
            csv_out_stages.append(stage)
            lambda_stage[key] > stage['input']

        self.run_pipeline(p)

        controls = log1_sqrt2_scale3(in_data, scale)

        for i, key in enumerate(out_keys):
            control = controls[i]
            if is_sa(control):
                control = np_sa_to_nd(control)[0]
            result = self._tmp_files.csv_read(
                        'out_{}.csv'.format(key), 
                        as_nd=True)
            self.assertTrue(np.allclose(control, result))

    def test_timify(self):
        in_file = path_of_data('with_dates.csv')

        p = Pipeline()

        csv_in = p.add(CSVRead(in_file))

        timify = p.add(Timify())
        csv_in['output'] > timify['input']

        np_out = p.add(NumpyWrite())
        timify['output'] > np_out['input']

        self.run_pipeline(p)
        result = np_out.get_stage().result

        ctrl_raw = csv_read(in_file)
        ctrl_dtype = np.dtype([(name, '<M8[D]') if 'dt' in name else 
                               (name, fmt) for name, fmt in 
                               ctrl_raw.dtype.descr])
        ctrl_better = csv_read(in_file, dtype=ctrl_dtype)

        self.assertEqual(result.dtype, ctrl_better.dtype)
        self.assertTrue(np.array_equal(result, ctrl_better))

    def test_identity(self):
        trials = [(('input0', 'input1'), ('output0', 'output1'), 
                   {'input0': 'output0', 'input1': 'output1'},
                   True),
                  (('input0', 'input1', 'input2'), 
                   ('input0_out', 'input1_out', 'input2_out'), 
                   ('input0', 'input1', 'input2'),
                   True),
                  (('input0', 'input1'), ('output0', 'output1'), 
                   {'output0': 'input0', 'output1': 'input1'},
                   False),
                  (('output0_in', 'output1_in', 'output2_in'),
                   ('output0', 'output1', 'output2'),
                   ('output0', 'output1', 'output2'),
                   False)]
        
        for input_keys, output_keys, arg, specify_input in trials:

            in_data_arrays = []
            out_nodes = []

            p = Pipeline()

            if specify_input:
                node_id = p.add(Identity(arg))
            else:
                node_id = p.add(Identity(output_keys=arg))

            for input_key, output_key, in zip(input_keys, output_keys):

                in_data = np_nd_to_sa(np.random.random((100, 10)))
                node_in = p.add(NumpyRead(in_data))
                node_in['output'] > node_id[input_key]

                node_out = p.add(NumpyWrite())
                node_id[output_key] > node_out['input']

                in_data_arrays.append(in_data)
                out_nodes.append(node_out)

            self.run_pipeline(p)

            for in_data, out_node in zip(in_data_arrays, out_nodes):
                self.assertTrue(np.array_equal(in_data, 
                                               out_node.get_stage().result))

    def test_apply_to_selected_cols(self):
        rows = 100
        cols = 10
        random_data = np.random.rand(rows, cols)
        # enough nans so that there /has/ to be a Nan in 1 of our 3 selected cols
        nans = 701
        with_nans = np.copy(random_data)
        for r, c in zip(np.random.randint(0, rows, nans), 
                        np.random.randint(0, cols, nans)):
            with_nans[r,c] = np.NaN
        trials = ((wrap('sklearn.preprocessing.StandardScaler'), 
                   (), 
                   'X_train', 
                   'X_new',
                   np_nd_to_sa(random_data)), 
                  (FillNA, 
                   (0,), 
                   'input', 
                   'output',
                   np_nd_to_sa(with_nans)))
        sel_cols = ('f2', 'f3', 'f4')
        trials = trials[1:]

        for trans_cls, args, in_key, out_key, in_data in trials:
            p = Pipeline()

            node_in = p.add(NumpyRead(in_data))
            node_selected = p.add(
                ApplyToSelectedCols(sel_cols, trans_cls, *args))
            node_in['output'] > node_selected[in_key]

            node_out = p.add(NumpyWrite())
            node_selected[out_key] > node_out['input']

            node_ctrl_split = p.add(SplitColumns(sel_cols))
            node_in['output'] > node_ctrl_split['input']

            node_ctrl_trans = p.add(trans_cls(*args))
            node_ctrl_split['output'] > node_ctrl_trans[in_key]

            node_ctrl_out = p.add(NumpyWrite())
            node_ctrl_trans[out_key] > node_ctrl_out['input']

            self.run_pipeline(p)

            result = node_out.get_stage().result
            ctrl = node_ctrl_out.get_stage().result

            for col in in_data.dtype.names:
                if col in sel_cols:
                    self.assertTrue(np.allclose(result[col], ctrl[col]))
                else:
                    self.assertTrue(np.allclose(
                        np.nan_to_num(result[col]), 
                        np.nan_to_num(in_data[col])))

    def test_merge(self):
        a1 = np.array([(0, 'Lisa', 2),
                       (1, 'Bill', 1),
                       (2, 'Fred', 2),
                       (3, 'Samantha', 2),
                       (4, 'Augustine', 1),
                       (5, 'William', 0)], dtype=[('id', int),
                                                  ('name', 'S64'),
                                                  ('dept_id', int)])
        a2 = np.array([(0, 'accts receivable'),
                       (1, 'accts payable'),
                       (2, 'shipping')], dtype=[('id', int),
                                                ('name', 'S64')])
        kwargs = {}

        p = Pipeline()
        a1_in = p.add(NumpyRead(a1))
        a2_in = p.add(NumpyRead(a2))
        merge = p.add(Merge('dept_id', 'id', **kwargs))
        out = p.add(NumpyWrite())

        out(merge(a1_in, a2_in))

        self.run_pipeline(p)

        result =  out.get_stage().result
        ctrl = obj_to_str(
                pd.DataFrame(a1).merge(
                    pd.DataFrame(a2),
                    left_on='dept_id',
                    right_on='id').to_records(index=False))

        assert(np.array_equal(result, ctrl))

    def test_split_by_inds(self):
        in_data = np.array(
            [(0, 0), (1, 1), (2, 0), (3, 1)], 
            dtype=[('id', int), ('include', int)])

        p = Pipeline()

        np_in = p.add(NumpyRead(in_data))

        query = p.add(Query('include != 0'))
        query(np_in)

        split_inds = p.add(SplitByInds())
        split_inds(np_in, query['output_inds'])

        out = p.add(NumpyWrite())
        out(split_inds)
        self.run_pipeline(p)

        ctrl = np.array(
            [(1, 1), (3, 1)], 
            dtype=[('id', int), ('include', int)])

        self.assertTrue(np.array_equal(ctrl, out.get_stage().result))

    def test_hstack(self):
        a = np.array(
                [(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)], 
                dtype=[('f0', float), ('f1', float)])
        b = np.array(
                [(0.2, 0.3), (1.2, 1.3), (2.2, 2.3)], 
                dtype=[('f2', float), ('f3', float)])
        ctrl = np.array(
                [(0.0, 0.1, 0.2, 0.3), (1.0, 1.1, 1.2, 1.3), 
                 (2.0, 2.1, 2.2, 2.3)], 
                dtype=[('f0', float), ('f1', float), ('f2', float), 
                       ('f3', float)])

        p = Pipeline()

        np_in_a = p.add(NumpyRead(a))

        np_in_b = p.add(NumpyRead(b))

        hstack = p.add(HStack(2))
        hstack(np_in_a, np_in_b)

        out = p.add(NumpyWrite())

        out(hstack)

        p.run()

        self.assertTrue(np.array_equal(ctrl, out.get_stage().result))

if __name__ == '__main__':
    unittest.main()
