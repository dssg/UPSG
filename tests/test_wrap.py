import numpy as np
from os import system
import unittest
from inspect import getargspec

from sklearn.cross_validation import train_test_split

from upsg.wrap.wrap_sklearn import wrap, wrap_instance
from upsg.uobject import UObject, UObjectPhase
from upsg.pipeline import Pipeline
from upsg.fetch.csv import CSVRead
from upsg.export.csv import CSVWrite
from upsg.transform.split import SplitColumn, SplitTrainTest
from upsg.utils import np_nd_to_sa, np_sa_to_nd

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
            out_key, sk_method):

        infile_name = path_of_data(csv)

        stage0 = CSVRead(infile_name)
        stage1 = SplitColumn(-1)
        stage2 = SplitTrainTest(random_state = 0)
        stage3 = SplitTrainTest(random_state = 0)
        wrapped_sk_cls = wrap(sk_cls) 
        stage4 = wrapped_sk_cls(*init_args, **init_kwargs)
        stage5 = CSVWrite(outfile_name)

        p = Pipeline()

        uids = map(p.add, [stage0, stage1, stage2, stage3, stage4, stage5])

        p.connect(uids[0], 'out', uids[1], 'in')
        p.connect(uids[1], 'X', uids[2], 'in')
        p.connect(uids[1], 'y', uids[3], 'in')
        input_keys = stage4.input_keys
        if 'X_train' in input_keys:
            p.connect(uids[2], 'train', uids[4], 'X_train')
        if 'X_test' in input_keys:
            p.connect(uids[2], 'test', uids[4], 'X_test')
        if 'y_train' in input_keys:
            p.connect(uids[3], 'train', uids[4], 'y_train')
        if 'y_test' in input_keys:
            p.connect(uids[3], 'test', uids[4], 'y_test')
        p.connect(uids[4], out_key, uids[5], 'in')

        p.run()

        ctrl_sk_inst = sk_cls(*init_args, **init_kwargs)
        ctrl_in_sa = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        ctrl_in_nd, ctrl_in_sa_dtype = np_sa_to_nd(ctrl_in_sa)
        ctrl_y_nd = ctrl_in_nd[:,-1]
        ctrl_X_nd = ctrl_in_nd[:,:-1]
        ctrl_X_train, ctrl_X_test, ctrl_y_train, ctrl_y_test = (
            train_test_split(ctrl_X_nd, ctrl_y_nd, random_state = 0))
        ctrl_sk_inst.fit(ctrl_X_train, ctrl_y_train)
        ctrl_method = getattr(ctrl_sk_inst, sk_method) 
        #TODO this is hacky. Find a nicer method to decide what arguments we
        #   put in.
        if sk_method == 'transform':
            control = ctrl_method(ctrl_X_train)
        elif sk_method == 'predict':
            control = ctrl_method(ctrl_X_test)
        else:
            control = ctrl_method(ctrl_X_test, ctrl_y_test)

        result = np.genfromtxt(outfile_name, dtype=None, delimiter=',',
            names=True).view(dtype = control.dtype).reshape(control.shape)

        self.assertTrue(np.array_equal(result, control) or 
            np.allclose(result, control))
    def test_transform(self):
        from sklearn.preprocessing import Imputer
        self.__simple_pipeline('missing_vals.csv', Imputer, (), 
            {'strategy' : 'mean', 'missing_values' : 'NaN'}, 'X_new', 'transform')
    def test_predict(self):
        from sklearn.svm import SVC
        self.__simple_pipeline('numbers.csv', SVC, (), {}, 'y_pred', 'predict')
    def tearDown(self):
        system('rm *.upsg')
        system('rm {}'.format(outfile_name))

if __name__ == '__main__':
    unittest.main()
