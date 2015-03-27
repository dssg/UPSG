import numpy as np
from os import system
import unittest

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.transform.rename_cols import RenameCols

from utils import path_of_data, UPSGTestCase


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

if __name__ == '__main__':
    unittest.main()
