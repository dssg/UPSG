import numpy as np
from os import system
import unittest

from sklearn.cross_validation import KFold as SKKFold

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.fetch.np import NumpyRead
from upsg.transform.rename_cols import RenameCols
from upsg.transform.sql import RunSQL
from upsg.transform.split import Query, SplitColumns, KFold
from upsg.transform.fill_na import FillNA
from upsg.transform.label_encode import LabelEncode
from upsg.utils import np_nd_to_sa

from utils import path_of_data, UPSGTestCase, csv_read


class TestTransform(UPSGTestCase):

    def test_rename_cols(self):
        infile_name = path_of_data('mixed_csv.csv')
        rename_dict = {'name': 'designation', 'height': 'tallness'}

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        trans_node = p.add(RenameCols(rename_dict))
        csv_write_node = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_read_node['out'] > trans_node['in']
        trans_node['out'] > csv_write_node['in']

        p.run()

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
        join['joined'] > csv_out['in']

        p.run()

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('test_transform_test_sql_ctrl.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

    def test_split_columns(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('numbers.csv')))
        split = p.add(SplitColumns(('F1', 'F3')))
        csv_out_sel = p.add(CSVWrite(self._tmp_files('out_sel.csv')))
        csv_out_rest = p.add(CSVWrite(self._tmp_files('out_rest.csv')))

        csv_in['out'] > split['in']
        split['selected'] > csv_out_sel['in']
        split['rest'] > csv_out_rest['in']

        p.run()
        
        result = self._tmp_files.csv_read('out_sel.csv')
        ctrl = csv_read(path_of_data('test_split_columns_ctrl_selected.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

        result = self._tmp_files.csv_read('out_rest.csv')
        ctrl = csv_read(path_of_data('test_split_columns_ctrl_rest.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

    def test_query(self):
        # Make sure we can support simple queries
        q = Query("id < 10")
        ctrl = "in_table['id'] < 10".replace(' ', '')
        result = q.parse_query(('id', 'name')).replace(' ', '')
        self.assertEqual(result, ctrl)

        q = Query("name == 'Bruce'")
        ctrl = "in_table['name'] == 'Bruce'".replace(' ', '')
        result = q.parse_query(('id', 'name')).replace(' ', '')
        self.assertEqual(result, ctrl)

#        # Make sure we can parse fairly complex queries
#        # (Not yet supported)
#        q1 = Query("(id < 10) or (name == 'Bruce' and hired_dt != stop_dt)")
#        ctrl = ("(in_table ['id']<10 )or (in_table ['name']=='Bruce'and"
#                " in_table ['hired_dt']!=in_table ['stop_dt'])")
#        result = q1.parse_query(('id', 'name', 'hired_dt', 'stop_dt'))
#        self.assertEqual(result, ctrl)

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('query.csv')))
        q1_node = p.add(Query("id == value"))
        q2_node = p.add(Query("use_this_col == 'yes'"))
        q3_node = p.add(Query("value != 1"))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_in['out'] > q1_node['in']
        q1_node['out'] > q2_node['in']
        q2_node['out'] > q3_node['in']
        q3_node['out'] > csv_out['in']

        p.run()

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('query_ctrl.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

    def test_fill_na(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('missing_vals_mixed.csv')))
        fill_na = p.add(FillNA(-1))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_in['out'] > fill_na['in']
        fill_na['out'] > csv_out['in']

        p.run()

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('test_transform_test_fill_na_ctrl.csv'))
        
        self.assertTrue(np.array_equal(result, ctrl))

    def test_label_encode(self):

        p = Pipeline()

        csv_in = p.add(CSVRead(path_of_data('categories.csv')))
        le = p.add(LabelEncode())
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        csv_in['out'] > le['in']
        le['out'] > csv_out['in']

        p.run()

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
        np_in_X['out'] > kfold['in0']
        np_in_y['out'] > kfold['in1']

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
                    kfold[out_key] > stage['in']
                    slice_inds = train_test_inds[select_i]
                    expected_folds.append(
                            np_nd_to_sa(arrays[array_i][slice_inds]))

        p.run()

        for out_file, expected_fold in zip(out_files, expected_folds):
            self.assertTrue(np.array_equal(
                self._tmp_files.csv_read(out_file),
                expected_fold))

if __name__ == '__main__':
    unittest.main()
