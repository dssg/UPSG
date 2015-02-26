import numpy as np
from os import system
import unittest

from upsg.wrap_sklearn.wrap import wrap, wrap_instance
from upsg.uobject import UObject, UObjectPhase
from upsg.pipeline import Pipeline
from upsg.fetch.csv import CSVRead
from upsg.export.csv import CSVWrite
from upsg.transform.split import SplitColumn

from utils import path_of_data

outfile_name = path_of_data('_out.csv')

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
    def __simple_pipeline(self, csv, sk_cls, init_args, init_kwargs, 
            in_key_dict, out_key, sk_method):

        infile_name = path_of_data(csv)

        stage1 = CSVRead(infile_name)
        stage2 = SplitColumn(-1)
        wrapped_sk_cls = wrap(sk_cls) 
        stage3 = wrapped_sk_cls(*init_args, **init_kwargs)
        stage4 = CSVWrite(outfile_name)

        p = Pipeline()

        uid1 = p.add(stage1)
        uid2 = p.add(stage2)
        uid3 = p.add(stage3)
        uid4 = p.add(stage4)

        p.connect(uid1, 'out', uid2, 'in')
        map(lambda keys: p.connect(uid2, keys[0], uid3, keys[1]), 
            in_key_dict.items())
        p.connect(uid3, out_key, uid4, 'in')

        p.run()

        ctrl_sk_inst = sk_cls(*init_args, **init_kwargs)
        ctrl_in_sa = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        ctrl_in_nd = ctrl_in_sa.view(dtype=ctrl_in_sa[0][0].dtype).reshape(
            len(ctrl_in_sa), -1)
        ctrl_y_nd = ctrl_in_nd[:,-1]
        ctrl_X_nd = ctrl_in_nd[:,:-1]
        ctrl_sk_inst.fit(ctrl_X_nd, ctrl_y_nd)
        control = getattr(ctrl_sk_inst, sk_method)(ctrl_X_nd)

        result = np.genfromtxt(outfile_name, dtype=None, delimiter=',',
            names=True).view(dtype = control.dtype).reshape(control.shape)

        self.assertTrue(np.array_equal(result, control) or 
            np.allclose(result, control))
    def test_transform(self):
        from sklearn.preprocessing import Imputer
        self.__simple_pipeline('missing_vals.csv', Imputer, (), 
            {'strategy' : 'mean', 'missing_values' : 'NaN'}, 
            {'X' : 'X_train', 'y' : 'y_train'}, 
            'X_new', 'transform')
    def tearDown(self):
        system('rm *.upsg')
        system('rm {}'.format(outfile_name))

if __name__ == '__main__':
    unittest.main()
