import numpy as np
from os import system
import unittest

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.transform.rename_cols import RenameCols
from upsg.transform.sql import RunSQL

from utils import path_of_data, UPSGTestCase, csv_read


class TestTransform(UPSGTestCase):

    def xtest_rename_cols(self):
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
        # will be unhappy
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
        get_emp = p.add(RunSQL(q_sel_employees, [], ['tmp_emp'], db_url, {}))
        get_hrs = p.add(RunSQL(q_sel_hours, [], ['tmp_hrs'], db_url, {}))
        join = p.add(RunSQL(q_join, ['tmp_emp', 'tmp_hrs'], ['joined'],
                            db_url, {}))
        csv_out = p.add(CSVWrite(self._tmp_files('out.csv')))

        get_emp['tmp_emp'] > join['tmp_emp']
        get_hrs['tmp_hrs'] > join['tmp_hrs']
        join['joined'] > csv_out['in']

        p.run(verbose = True)

        result = self._tmp_files.csv_read('out.csv')
        ctrl = csv_read(path_of_data('test_transform_test_sql_ctrl.csv'))

        self.assertTrue(np.array_equal(result, ctrl))

if __name__ == '__main__':
    unittest.main()
