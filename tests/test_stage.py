import numpy as np
import unittest
import importlib
import pickle
from os import system
from StringIO import StringIO

from utils import UPSG_PATH, path_of_data, UPSGTestCase

from upsg.uobject import UObject, UObjectPhase
from upsg.wrap.wrap_sklearn import wrap

class TestStage(UPSGTestCase):

    def __pickle(self, cls, *module_args, **module_kwargs):
        if isinstance(cls, str): # cls is the name of the class
            split_import_path = cls.split('.')
            cls_name = split_import_path[-1]
            mod_name = '.'.join(split_import_path[:-1])
            mod = importlib.import_module(mod_name)
            stage = getattr(mod, cls_name)(*module_args, **module_kwargs)
        else: # cls is the class
            stage = cls(*module_args, **module_kwargs)
        pickle_target = StringIO()
        pickle.dump(stage, pickle_target)
        pickle_target.seek(0)
        stage_recovered = pickle.load(pickle_target)
        self.assertEqual(stage.input_keys, stage_recovered.input_keys)
        self.assertEqual(stage.output_keys, stage_recovered.output_keys)

    def test_pickle(self):
    # TODO this just makes sure the object can be pickled. It doesn't verify
    #   that the unpickled object is correct
        uo = UObject(UObjectPhase.Write)
        np_array = np.array([[0]])
        uo.from_np(np_array)
        self.__pickle('upsg.export.csv.CSVWrite',  path_of_data('_out.csv'))
        self.__pickle('upsg.fetch.csv.CSVRead',  path_of_data('mixed_csv.csv'))
        self.__pickle('upsg.fetch.np.NumpyRead',  np.array([[0]]))
        self.__pickle('upsg.transform.split.SplitTrainTest')
        self.__pickle('upsg.transform.split.SplitColumn', 0)
        self.__pickle('upsg.transform.rename_cols.RenameCols', 
            {'name' : 'rename'})
        self.__pickle(wrap('sklearn.preprocessing.Imputer'), strategy = 'mean', 
            missing_values = 'NaN')
        self.__pickle(wrap('sklearn.svm.SVC'), gamma = 0.1)

if __name__ == '__main__':
    unittest.main()

