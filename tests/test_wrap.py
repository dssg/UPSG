import numpy as np
from os import system
import unittest

from upsg.wrap_sklearn.wrap import wrap, wrap_instance
from upsg.uobject import UObject, UObjectPhase

from utils import path_of_data


class TestWrap(unittest.TestCase):
    def test_from_module(self):
        from sklearn.preprocessing import Imputer
        WrappedImputer = wrap(Imputer) 
        impute_stage = WrappedImputer()
    def test_from_string(self):
        WrappedImputer = wrap('sklearn.preprocessing.Imputer') 
        impute_stage = WrappedImputer()
    def test_wrap_instance(self):
        impute_stage = wrap_instance('sklearn.preprocessing.Imputer',
            strategy='median') 
        params = impute_stage.get_sklearn_instance().get_params()
        self.assertEqual(params['strategy'], 'median')
    def test_simple_use(self):
        from sklearn.preprocessing import Imputer
        WrappedImputer = wrap(Imputer) 
        impute_stage = WrappedImputer(strategy='mean', missing_values='NaN', axis = 0)

        filename = path_of_data('missing_vals.csv')
        uo_in = UObject(UObjectPhase.Write)
        uo_in.from_csv(filename)
        uo_in.to_read_phase()

        ctrl_imputer = Imputer()
        ctrl_X_sa = np.genfromtxt(filename, dtype=None, delimiter=",", 
            names=True)
        ctrl_X_nd = ctrl_X_sa.view(dtype=ctrl_X_sa[0][0].dtype).reshape(
            len(ctrl_X_sa), -1)
        ctrl_X_new_nd = ctrl_imputer.fit_transform(ctrl_X_nd)
        control = ctrl_X_new_nd.view(dtype=ctrl_X_sa.dtype).reshape(
                    len(ctrl_X_new_nd))

        uo_out = impute_stage.run(['X_new'], X=uo_in)['X_new']
        uo_out.to_read_phase()
        result = uo_out.to_np()

        self.assertTrue(np.array_equal(result, control))
    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()
