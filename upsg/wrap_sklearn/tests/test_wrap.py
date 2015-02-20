from upsg.wrap_sklearn.wrap import wrap
from upsg.uobject import UObject, UObjectPhase

import numpy as np
from os import system
import unittest

class TestWrap(unittest.TestCase):
    def test_from_module(self):
        from sklearn.preprocessing import Imputer
        WrappedImputer = wrap(Imputer) 
        impute_stage = WrappedImputer()
    def test_from_string(self):
        WrappedImputer = wrap('sklearn.preprocessing.Imputer') 
        impute_stage = WrappedImputer()
    def test_simple_use(self):
        from sklearn.preprocessing import Imputer
        WrappedImputer = wrap(Imputer) 
        impute_stage = WrappedImputer(strategy='mean', missing_values='NaN', axis = 0)

        filename = 'missing_vals.csv'
        uo_in = UObject(UObjectPhase.Write)
        uo_in.from_csv(filename)
        uo_in.to_read_phase()

        ctrl_imputer = Imputer()
        ctrl_X_sa = np.genfromtxt(filename, dtype=None, delimiter=",", 
            names=True)
        ctrl_X = ctrl_X_sa.view(dtype=ctrl_X_sa[0][0].dtype).reshape(
            len(ctrl_X_sa), -1)
        control_sa = ctrl_imputer.fit_transform(ctrl_X)
        control = control_sa.view(dtype=ctrl_X_sa.dtype).reshape(
                    len(ctrl_X_sa))

        uo_out = impute_stage.run(X=uo_in)['X_new']
        uo_out.to_read_phase()
        result = uo_out.to_np()


        self.assertTrue(np.array_equal(result, control))
    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()
