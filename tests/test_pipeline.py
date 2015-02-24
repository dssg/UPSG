import numpy as np
from os import system
import unittest

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.wrap_sklearn.wrap import wrap_instance
from utils import path_of_data

outfile_name = path_of_data('_out.csv')

class TestUObject(unittest.TestCase):
    def test_rw(self):
        infile_name = path_of_data('mixed_csv.csv')        

        p = Pipeline()

        csv_read_uid = p.add(CSVRead(infile_name))
        csv_write_uid = p.add(CSVWrite(outfile_name))

        p.connect(csv_read_uid, 'out', csv_write_uid, 'in')

        p.run()

        control = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        result = np.genfromtxt(outfile_name, dtype=None, delimiter=",", 
            names=True)
        
        self.assertTrue(np.array_equal(result, control))

    def test_3_stage(self):
        from sklearn.preprocessing import Imputer
    
        infile_name = path_of_data('missing_vals.csv')        

        p = Pipeline()

        csv_read_uid = p.add(CSVRead(infile_name))
        csv_write_uid = p.add(CSVWrite(outfile_name))
        impute_uid = p.add(wrap_instance(Imputer))

        p.connect(csv_read_uid, 'out', impute_uid, 'X')
        p.connect(impute_uid, 'X_new', csv_write_uid, 'in')

        p.run()

        ctrl_imputer = Imputer()
        ctrl_X_sa = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        num_type = ctrl_X_sa[0][0].dtype
        ctrl_X_nd = ctrl_X_sa.view(dtype = num_type).reshape(
            len(ctrl_X_sa), -1)
        ctrl_X_new_nd = ctrl_imputer.fit_transform(ctrl_X_nd)
        control = ctrl_X_new_nd

        res_sa = np.genfromtxt(outfile_name, dtype=None, delimiter=",", 
            names=True)
        result = res_sa.view(dtype = num_type).reshape(len(res_sa), -1)
        
        print result
        print control
        self.assertTrue(np.allclose(result, control))

    def tearDown(self):
        system('rm *.upsg')
        system('rm {}'.format(outfile_name))

if __name__ == '__main__':
    unittest.main()
