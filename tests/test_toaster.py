import os
import unittest

from upsg.toaster import DataToaster

from utils import path_of_data, UPSGTestCase

class TestToaster(UPSGTestCase):
    def test_toaster(self):
        dt = DataToaster()
        # Read in a csv
        dt.from_csv(path_of_data('test_toaster.csv'))
        # Training is data before 2006-06-15; testing is after. The column
        # giving us classification is 'cat'
        dt.split_by_query('cat', "date < DT('2006-06-15')") 
        # Select features (manually, in this case)
        dt.transform_select_cols(('factor_1', 'factor_2'))
        # Do some last-minute cleanup
        dt.transform_with_sklearn('sklearn.preprocessing.StandardScaler')
        # Try a bunch of classifiers and parameters
        dt.classify_and_report(report_file_name=self._tmp_files('report.html'))
        dt.run()

        self.assertTrue(os.path.isfile(self._tmp_files('report.html')))

if __name__ == '__main__':
    unittest.main()
